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

# kAXValueCGPointType == 1, kAXValueCGSizeType == 2 (AXValue.h enum; not
# exported as symbols).
_KAX_VALUE_CGPOINT_TYPE = 1
_KAX_VALUE_CCGSIZE_TYPE = 2

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


class _CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


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


def element_size(element: Any) -> Optional[Tuple[float, float]]:
    """Read AXSize as (width, height); None when unreadable or not a size."""
    value = copy_attribute(element, "AXSize")
    if not isinstance(value, int) or not value:
        return None
    size = _CGSize()
    if not _hs.AXValueGetValue(value, _KAX_VALUE_CCGSIZE_TYPE, ctypes.byref(size)):
        return None
    return (float(size.width), float(size.height))


def _running_apps() -> List[Any]:
    from AppKit import NSWorkspace

    return NSWorkspace.sharedWorkspace().runningApplications()


def _pid_via_ps(name: str) -> Optional[int]:
    """Authoritative pid lookup that does not depend on NSWorkspace freshness.

    A process without a spinning runloop (the pipeline is plain Python) does
    not receive NSWorkspace updates for apps launched AFTER it started
    (observed C25: DB Browser launched by the readiness wait stayed invisible
    to runningApplications() for 30+s while fully booted). ``ps`` reflects the
    real process table. Matches the app binary path
    ``<App>.app/Contents/MacOS/<name>`` to avoid matching osascript arguments.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,comm="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return None
    needle = f"/MacOS/{name}"
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, comm = parts
        if comm.endswith(needle) or comm == name:
            return int(pid_s)
    return None


def app_pid_for_name(name: str) -> Optional[int]:
    # ps first: NSWorkspace.runningApplications() is stale in a process
    # without a spinning runloop — it kept returning the pre-relaunch pid for
    # an entire run after stage prep killed and relaunched DB Browser, so
    # every AX call targeted a dead pid (-25204) while System Events, which
    # resolves by name, kept working (C25 trace: pid 27913 returned 23 times
    # across the relaunch boundary).
    pid = _pid_via_ps(name)
    if pid is not None:
        return pid
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


def _window_title(window_element: Any) -> Optional[str]:
    try:
        title = copy_attribute(window_element, "AXTitle")
    except AxCallError:
        return None
    return title if isinstance(title, str) else None


def window_is_modal(window_element: Any) -> bool:
    """C33: True when the window is a modal dialog or sheet (AXDialog subrole
    or role, or AXSheet role) — never a valid editor-identity window."""
    try:
        subrole = copy_attribute(window_element, "AXSubrole")
    except AxCallError:
        subrole = None
    if subrole == "AXDialog":
        return True
    try:
        role = copy_attribute(window_element, "AXRole")
    except AxCallError:
        role = None
    return role in ("AXDialog", "AXSheet")


def _select_editor_windows(windows: List[Any], title_hints) -> List[Any]:
    """C33: keep only windows whose AXTitle contains a hint (the main window
    carries the app name and the .db filename; modal dialogs carry neither).
    Falls back to non-modal windows when no title matches, then to all."""
    matched = [
        w for w in windows
        if (t := _window_title(w)) is not None
        and any(h in t for h in title_hints)
    ]
    if matched:
        return matched
    non_modal = [w for w in windows if not window_is_modal(w)]
    return non_modal or list(windows)


def frontmost_modal(app_element: Any) -> Optional[Tuple[Any, str]]:
    """C33: (window_element, title) when the app's FRONTMOST window is a
    modal dialog/sheet, else None. AXWindows is returned front-to-back, so
    index 0 is the frontmost."""
    windows = copy_attribute(app_element, "AXWindows") or []
    if not windows:
        return None
    front = windows[0]
    if not window_is_modal(front):
        return None
    return (front, _window_title(front) or "")


def press_button(window_element: Any, title: str) -> bool:
    """C33: AXPress the AXButton whose AXTitle == title inside one window
    subtree. Returns False when no such button exists (the caller falls
    back, e.g. to Esc)."""
    stack: List[Any] = [window_element]
    visited = 0
    while stack:
        el = stack.pop()
        visited += 1
        if visited > _MAX_ELEMENTS:
            return False
        try:
            role = copy_attribute(el, "AXRole")
        except AxCallError:
            role = None
        if role == "AXButton":
            try:
                btn_title = copy_attribute(el, "AXTitle")
            except AxCallError:
                btn_title = None
            if btn_title == title:
                press(el)
                return True
        try:
            children = copy_attribute(el, "AXChildren") or []
        except AxCallError:
            children = []
        stack.extend(children)
    return False


def find_text_areas(
    app_element: Any, title_hints: Optional[Tuple[str, ...]] = None
) -> List[Tuple[Any, Optional[float]]]:
    """Traverse AXWindows/AXChildren from an application element and collect
    every AXTextArea with its vertical position (None when unreadable).

    Pure attribute reads — no System Events, no ``entire contents`` — which is
    exactly what survives ScreenCaptureKit capture (C21 probe). Raises
    AxCallError only when the top-level window read itself fails; per-element
    read errors are tolerated (element skipped).

    C33: when ``title_hints`` is given, only windows whose AXTitle contains a
    hint are traversed — editor identity must come from the MAIN window, never
    a modal dialog's field. With 'Edit table definition' frontmost, the
    dialog's read-only SQL preview is the top-most AXTextArea app-wide and
    poisons both the length read and the focus path (C32b halt).
    """
    windows = copy_attribute(app_element, "AXWindows") or []
    if title_hints:
        windows = _select_editor_windows(windows, title_hints)
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


def press_execute_tab(app_element: Any) -> bool:
    """Find the 'Execute SQL' AXRadioButton in any window and press it.

    A freshly opened DB Browser window sits on a tab with no AXTextArea, so
    the text-area enumeration cannot succeed until the editor's tab is shown
    (C25 readiness wait). Same attribute-read traversal as find_text_areas;
    returns True when the press action was invoked.
    """
    windows = copy_attribute(app_element, "AXWindows") or []
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
        if role == "AXRadioButton":
            try:
                title = copy_attribute(el, "AXTitle")
            except AxCallError:
                title = None
            if title == "Execute SQL":
                press(el)
                return True
        try:
            children = copy_attribute(el, "AXChildren") or []
        except AxCallError:
            children = []
        for child in children:
            stack.append((child, depth + 1))
    return False


def find_radio_buttons(
    app_element: Any, title_hints: Optional[Tuple[str, ...]] = None
) -> List[Tuple[Any, str]]:
    """Collect ``(element, title)`` for every AXRadioButton under the app's
    windows (the view tabs). Same attribute-read traversal as
    ``find_text_areas``; C33 window scoping applies when hints are given."""
    windows = copy_attribute(app_element, "AXWindows") or []
    if title_hints:
        windows = _select_editor_windows(windows, title_hints)
    found: List[Tuple[Any, str]] = []
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
        if role == "AXRadioButton":
            try:
                title = copy_attribute(el, "AXTitle")
            except AxCallError:
                title = None
            if title:
                found.append((el, title))
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
