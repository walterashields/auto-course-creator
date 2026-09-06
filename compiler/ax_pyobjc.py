#!/usr/bin/env python3
"""
compiler/ax_pyobjc.py

C22: direct Accessibility (AX) API access via ctypes on the HIServices
framework, bypassing System Events entirely.

Why: under ScreenCaptureKit capture (recording), the System Events
``entire contents`` enumeration returns a genuinely EMPTY subtree for DB
Browser's window (C21 probe: 0% success in conditions B/D, fast
~0.17s wsda-no-text-area). Attribute-level AX reads survive capture; only the
System Events enumeration path dies. This module performs the same discovery
with per-attribute AXUIElementCopyAttributeValue calls — the class of read the
C21 probe proved reliable — plus traversal of AXChildren arrays via
CoreFoundation CFArray calls.

Testability: every AX interaction goes through ``copy_attribute`` /
``set_focused`` / ``focused_element_info`` / ``find_text_areas``, which
operate on plain Python values (strings, lists, opaque element handles). Tests
monkeypatch those functions with a fake tree; no ctypes required.

Note: AXUIElementRef is NOT toll-free bridged to NSObject, so returned
element handles stay raw pointers (ints). CFString/CFArray values are
converted to Python str/list.
"""

from __future__ import annotations

import ctypes
from typing import Any, List, Optional, Tuple

_CORE_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_HS_PATH = (
    "/System/Library/Frameworks/ApplicationServices.framework/"
    "Frameworks/HIServices.framework/HIServices"
)

_cf = ctypes.CDLL(_CORE_PATH)
_hs = ctypes.CDLL(_HS_PATH)

# -- CoreFoundation ------------------------------------------------------------
_cf.CFStringCreateWithCString.restype = ctypes.c_void_p
_cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
_cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
_cf.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_cf.CFStringGetCString.restype = ctypes.c_bool
_cf.CFStringGetCString.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
]
_cf.CFStringGetLength.restype = ctypes.c_long
_cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
_cf.CFArrayGetCount.restype = ctypes.c_long
_cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
_cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
_cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
_cf.CFGetTypeID.restype = ctypes.c_ulong
_cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
_cf.CFStringGetTypeID.restype = ctypes.c_ulong
_cf.CFArrayGetTypeID.restype = ctypes.c_ulong

_kUTF8 = 0x08000100

# -- HIServices / AX -----------------------------------------------------------
_hs.AXUIElementCreateSystemWide.restype = ctypes.c_void_p
_hs.AXUIElementCreateSystemWide.argtypes = []
_hs.AXUIElementCreateApplication.restype = ctypes.c_void_p
_hs.AXUIElementCreateApplication.argtypes = [ctypes.c_int32]
_hs.AXUIElementCopyAttributeValue.restype = ctypes.c_int
_hs.AXUIElementCopyAttributeValue.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
]
_hs.AXUIElementSetAttributeValue.restype = ctypes.c_int
_hs.AXUIElementSetAttributeValue.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_hs.AXUIElementGetPid.restype = ctypes.c_int
_hs.AXUIElementGetPid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
_hs.AXUIElementPerformAction.restype = ctypes.c_int
_hs.AXUIElementPerformAction.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_hs.AXValueGetValue.restype = ctypes.c_bool
_hs.AXValueGetValue.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

# kAXValueCGPointType == 1 (AXValue.h enum; not exported as a symbol).
_KAX_VALUE_CGPOINT_TYPE = 1

_TRUE_PTR = ctypes.c_void_p.in_dll(_cf, "kCFBooleanTrue")

_ERR_NAMES = {
    -25200: "kAXErrorFailure",
    -25201: "kAXErrorIllegalArgument",
    -25202: "kAXErrorInvalidUIElement",
    -25203: "kAXErrorInvalidUIElementObserver",
    -25204: "kAXErrorCannotComplete",
    -25205: "kAXErrorAttributeUnsupported",
    -25206: "kAXErrorActionUnsupported",
    -25207: "kAXErrorNotificationUnsupported",
    -25208: "kAXErrorNotImplemented",
    -25211: "kAXErrorAPIDisabled",
    -25212: "kAXErrorNoValue",
    -25213: "kAXErrorParameterizedAttributeUnsupported",
    -25214: "kAXErrorNotEnoughPrecision",
}

# Attribute name -> CFStringRef (created once; CoreFoundation constants).
_ATTR_REFS = {}


def _attr_ref(name: str) -> int:
    ref = _ATTR_REFS.get(name)
    if ref is None:
        ref = _cf.CFStringCreateWithCString(None, name.encode(), _kUTF8)
        _ATTR_REFS[name] = ref
    return ref


class AxCallError(Exception):
    """An AX API call failed with a non-success AXError."""

    def __init__(self, operation: str, err: int):
        self.err = err
        super().__init__(f"{operation}: {err} ({error_name(err)})")


def error_name(err: int) -> str:
    return _ERR_NAMES.get(err, f"AXError {err}")


def _as_str(ptr: int) -> Optional[str]:
    if not ptr:
        return None
    fast = _cf.CFStringGetCStringPtr(ptr, _kUTF8)
    if fast:
        return fast.decode()
    n = _cf.CFStringGetLength(ptr) + 1
    buf = ctypes.create_string_buffer(n * 4)
    if _cf.CFStringGetCString(ptr, buf, len(buf), _kUTF8):
        return buf.value.decode()
    return None


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def create_system_wide() -> int:
    return _hs.AXUIElementCreateSystemWide()


def create_application(pid: int) -> int:
    return _hs.AXUIElementCreateApplication(int(pid))


def copy_attribute(element: Any, name: str) -> Any:
    """Read one AX attribute, converted to a plain Python value.

    Returns None for attributes with no value (kAXErrorNoValue /
    AttributeUnsupported / InvalidUIElement) — common mid-traversal. Raises
    AxCallError for real failures (CannotComplete, APIDisabled, ...).

    CFString values -> str; CFArray values -> list of raw element pointers;
    everything else (AXUIElementRef, AXValueRef, CFNumber) -> raw int.
    """
    out = ctypes.c_void_p()
    err = _hs.AXUIElementCopyAttributeValue(
        element, _attr_ref(name), ctypes.byref(out)
    )
    if err != 0:
        if err in (-25212, -25205, -25202):  # NoValue / Unsupported / Invalid
            return None
        raise AxCallError(f"copy {name}", err)
    if not out.value:
        return None
    ptr = out.value
    type_id = _cf.CFGetTypeID(ptr)
    if type_id == _cf.CFStringGetTypeID():
        return _as_str(ptr)
    if type_id == _cf.CFArrayGetTypeID():
        return [
            _cf.CFArrayGetValueAtIndex(ptr, i)
            for i in range(_cf.CFArrayGetCount(ptr))
        ]
    return ptr


def set_focused(element: Any) -> None:
    """Set AXFocused = true on an element."""
    err = _hs.AXUIElementSetAttributeValue(
        element, _attr_ref("AXFocused"), _TRUE_PTR
    )
    if err != 0:
        raise AxCallError("set AXFocused", err)


def press(element: Any, action: str = "AXPress") -> None:
    """Perform an AX action (default AXPress) on an element."""
    err = _hs.AXUIElementPerformAction(element, _attr_ref(action))
    if err != 0:
        raise AxCallError(f"perform {action}", err)


def get_pid(element: Any) -> int:
    pid = ctypes.c_int32(-1)
    err = _hs.AXUIElementGetPid(element, ctypes.byref(pid))
    if err != 0:
        raise AxCallError("get pid", err)
    return int(pid.value)


def element_position(element: Any) -> Optional[Tuple[float, float]]:
    """Read AXPosition as (x, y); None when unreadable or not a point."""
    value = copy_attribute(element, "AXPosition")
    if not isinstance(value, int) or not value:
        return None
    pt = _CGPoint()
    if not _hs.AXValueGetValue(value, _KAX_VALUE_CGPOINT_TYPE, ctypes.byref(pt)):
        return None
    return (float(pt.x), float(pt.y))


def _running_apps() -> List[Any]:
    from AppKit import NSWorkspace

    return NSWorkspace.sharedWorkspace().runningApplications()


def app_pid_for_name(name: str) -> Optional[int]:
    for app in _running_apps():
        if app.localizedName() == name:
            return int(app.processIdentifier())
    return None


def app_name_for_pid(pid: int) -> Optional[str]:
    for app in _running_apps():
        if int(app.processIdentifier()) == int(pid):
            return app.localizedName()
    return None


# C22 traversal guards: DB Browser's tree is a few hundred nodes; these caps
# only stop pathological runaway, not legitimate trees.
_MAX_DEPTH = 64
_MAX_ELEMENTS = 20000


def find_text_areas(app_element: Any) -> List[Tuple[Any, Optional[float]]]:
    """Traverse AXWindows/AXChildren from an application element and collect
    every AXTextArea with its vertical position (None when unreadable).

    Pure attribute reads — no System Events, no ``entire contents`` — which is
    exactly what survives ScreenCaptureKit capture (C21 probe). Raises
    AxCallError only when the top-level window read itself fails; per-element
    read errors are tolerated (element skipped).
    """
    windows = copy_attribute(app_element, "AXWindows") or []
    found: List[Tuple[Any, Optional[float]]] = []
    # Stack of (element, depth); windows first.
    stack: List[Tuple[Any, int]] = [(w, 0) for w in windows]
    visited = 0
    while stack:
        el, depth = stack.pop()
        visited += 1
        if visited > _MAX_ELEMENTS or depth > _MAX_DEPTH:
            continue
        try:
            role = copy_attribute(el, "AXRole")
        except AxCallError:
            role = None
        if role == "AXTextArea":
            try:
                pos = element_position(el)
            except AxCallError:
                pos = None
            found.append((el, pos[1] if pos else None))
        try:
            children = copy_attribute(el, "AXChildren") or []
        except AxCallError:
            children = []
        for child in children:
            stack.append((child, depth + 1))
    return found


def focused_element_info(app_element: Any) -> Optional[dict]:
    """Read the focused element of one application element (AXFocusedUIElement
    supported on application elements).

    The system-wide AXFocusedUIElement read is NOT used: from this process it
    consistently returns kAXErrorCannotComplete (-25204) even while all other
    AX reads succeed. Returns {"element", "pid", "role"} or None when the app
    reports no focused element.
    """
    focused = copy_attribute(app_element, "AXFocusedUIElement")
    if not focused:
        return None
    try:
        pid = get_pid(focused)
    except AxCallError:
        pid = None
    try:
        role = copy_attribute(focused, "AXRole")
    except AxCallError:
        role = None
    return {"element": focused, "pid": pid, "role": role}
