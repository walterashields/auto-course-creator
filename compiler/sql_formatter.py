#!/usr/bin/env python3
"""
compiler/sql_formatter.py

Format SQL queries to match the SQL Essentials course standard:
- Comment block before every query
- SQL keywords in ALL CAPS
- One clause per line
- Each SELECT field on its own indented line with trailing commas
- AS for all aliases
- Table aliases for joins
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple


# Keywords that must appear in ALL CAPS.
_SINGLE_KEYWORDS = {
    "select", "from", "where", "join", "on", "as", "in", "between", "like",
    "distinct", "count", "sum", "avg", "min", "max", "round", "case", "when",
    "then", "else", "end", "limit", "having", "by", "group", "order", "inner",
    "left", "right", "full", "outer", "cross", "natural", "union", "all",
    "is", "null", "not", "or", "and", "asc", "desc", "coalesce",
}

# Phrases of two tokens that must be uppercased together.
_TWO_WORD_KEYWORDS = {
    ("group", "by"),
    ("order", "by"),
    ("inner", "join"),
    ("left", "join"),
    ("right", "join"),
    ("full", "outer"),
    ("outer", "join"),
    ("cross", "join"),
    ("natural", "join"),
}

_JOIN_KEYWORDS = {"JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN", "CROSS JOIN", "NATURAL JOIN"}


class SQLFormatError(ValueError):
    """Raised when a query cannot be formatted reliably."""


def _today() -> str:
    return datetime.now().strftime("%m/%d/%Y")


def _normalize_whitespace(query: str) -> str:
    """Collapse runs of whitespace into a single space and strip."""
    return re.sub(r"\s+", " ", query).strip()


def _uppercase_keywords(query: str) -> str:
    """Uppercase SQL keywords while preserving quoted string literals."""
    result: List[str] = []
    i = 0
    n = len(query)
    in_string = False
    string_char = ""
    while i < n:
        ch = query[i]
        if ch in ("'", '"'):
            if not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char:
                # Check for escaped quote.
                if i + 1 < n and query[i + 1] == string_char:
                    result.append(ch)
                    i += 1
                else:
                    in_string = False
                    string_char = ""
            result.append(ch)
            i += 1
            continue

        if in_string:
            result.append(ch)
            i += 1
            continue

        # Try two-word keyword.
        if i + 1 < n and query[i + 1].isalpha():
            rest = query[i:]
            match = re.match(r"([a-zA-Z]+)\s+([a-zA-Z]+)", rest)
            if match:
                w1, w2 = match.group(1).lower(), match.group(2).lower()
                if (w1, w2) in _TWO_WORD_KEYWORDS:
                    result.append(f"{w1.upper()} {w2.upper()}")
                    i += len(match.group(0))
                    continue

        # Single-word keyword.
        word_match = re.match(r"[a-zA-Z]+", query[i:])
        if word_match:
            word = word_match.group(0)
            if word.lower() in _SINGLE_KEYWORDS:
                result.append(word.upper())
            else:
                result.append(word)
            i += len(word)
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _ensure_semicolon(query: str) -> str:
    query = query.rstrip()
    if not query.endswith(";"):
        query += ";"
    return query


def _split_top_level(text: str, delimiter: str) -> List[str]:
    """Split `text` by `delimiter` commas at the top level (outside parens/strings)."""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    in_string = False
    string_char = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in ("'", '"'):
            if not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char:
                if i + 1 < len(text) and text[i + 1] == string_char:
                    current.append(ch)
                    i += 1
                else:
                    in_string = False
        elif not in_string and ch == "(":
            depth += 1
        elif not in_string and ch == ")":
            depth -= 1
        elif not in_string and depth == 0 and ch == delimiter:
            parts.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    if current or text.endswith(delimiter):
        parts.append("".join(current).strip())
    return parts


def _ensure_alias(field: str) -> str:
    """Add AS to a field alias if it is missing."""
    field = field.strip()
    if not field:
        return field
    # If the field already ends with AS "..." or AS ..., leave it.
    if re.search(r"\s+AS\s+['\"A-Za-z_][A-Za-z0-9_'\"]*$", field, re.IGNORECASE):
        return field
    # If the last token looks like an alias (identifier after expression), add AS.
    # e.g. "SUM(amount) total" -> "SUM(amount) AS total"
    match = re.match(r"(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)$", field)
    if match:
        body, alias = match.group(1).strip(), match.group(2)
        if alias.upper() not in _SINGLE_KEYWORDS and alias.upper() not in {"FROM", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING"}:
            return f"{body} AS {alias}"
    return field


def _format_select_fields(select_clause: str) -> List[str]:
    """Return each SELECT field as an indented line."""
    fields = _split_top_level(select_clause, ",")
    formatted: List[str] = []
    for idx, field in enumerate(fields):
        field = _ensure_alias(field.strip())
        comma = "," if idx < len(fields) - 1 else ""
        formatted.append(f"    {field}{comma}")
    return formatted


def _find_clauses(query: str) -> List[Tuple[str, str]]:
    """
    Parse a normalized uppercased query into (clause_keyword, content) pairs.
    Returns e.g. [("SELECT", "*"), ("FROM", "Orders"), ("WHERE", "status = 'Shipped'")].
    """
    # Tokenize into keyword positions. Keywords are already uppercased.
    pattern = re.compile(
        r"\b(SELECT|FROM|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL OUTER JOIN|CROSS JOIN|NATURAL JOIN|JOIN|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT)\b"
    )
    tokens = [(m.group(1), m.start()) for m in pattern.finditer(query)]

    clauses: List[Tuple[str, str]] = []
    for i, (keyword, start) in enumerate(tokens):
        content_start = start + len(keyword)
        content_end = tokens[i + 1][1] if i + 1 < len(tokens) else len(query)
        content = query[content_start:content_end].strip()
        # Remove trailing semicolon from content; we re-add it later.
        content = content.rstrip(";").strip()
        clauses.append((keyword, content))
    return clauses


def _make_alias(table: str, used: set) -> str:
    """Generate a short table alias that has not been used yet."""
    base = table[0].lower()
    if base not in used:
        return base
    for length in range(2, len(table) + 1):
        candidate = table[:length].lower()
        if candidate not in used:
            return candidate
    # Fallback: append numbers.
    counter = 1
    while f"{base}{counter}" in used:
        counter += 1
    return f"{base}{counter}"


def _parse_table_with_alias(text: str, table_aliases: dict, used_aliases: set) -> str:
    """Return 'Table AS alias' for a table reference, adding alias if missing."""
    text = text.strip()
    # Already has AS alias?
    m = re.match(r"(\S+)\s+AS\s+(\S+)$", text, re.IGNORECASE)
    if m:
        table = m.group(1)
        alias = m.group(2)
        table_aliases[table] = alias
        used_aliases.add(alias.lower())
        return text
    # Has bare alias (identifier after table) e.g. "Orders o"?
    parts = text.split()
    if len(parts) == 2 and parts[1].lower() not in _SINGLE_KEYWORDS:
        table, alias = parts[0], parts[1]
        table_aliases[table] = alias
        used_aliases.add(alias.lower())
        return f"{table} AS {alias}"
    # No alias: if table already has an assigned alias, reuse it.
    table = parts[0]
    if table in table_aliases:
        return f"{table} AS {table_aliases[table]}"
    alias = _make_alias(table, used_aliases)
    table_aliases[table] = alias
    used_aliases.add(alias.lower())
    return f"{table} AS {alias}"


def _rewrite_with_aliases(text: str, table_aliases: dict) -> str:
    """Replace table names with aliases in a condition/expression."""
    # Sort by length descending so longer table names replace first.
    for table in sorted(table_aliases, key=len, reverse=True):
        alias = table_aliases[table]
        text = re.sub(rf"\b{re.escape(table)}\.", f"{alias}.", text)
    return text


def _format_from_and_joins(clauses: List[Tuple[str, str]], table_aliases: dict) -> List[str]:
    """Reconstruct FROM and JOIN clauses with aliases."""
    lines: List[str] = []
    used_aliases: set = set(alias.lower() for alias in table_aliases.values())
    join_clauses: List[Tuple[str, str]] = []
    has_joins = any(k in _JOIN_KEYWORDS for k, _ in clauses)

    for keyword, content in clauses:
        if keyword == "FROM":
            if has_joins:
                from_part = _parse_table_with_alias(content, table_aliases, used_aliases)
                lines.append(f"FROM {from_part}")
            else:
                lines.append(f"FROM {content}")
        elif keyword in _JOIN_KEYWORDS:
            join_clauses.append((keyword, content))

    for keyword, content in join_clauses:
        # Content is like "Orders ON Customers.customer_id = Orders.customer_id".
        on_match = re.match(r"(.+?)\s+ON\s+(.+)", content, re.IGNORECASE)
        if on_match:
            table_part = on_match.group(1).strip()
            on_condition = on_match.group(2).strip()
            table_ref = _parse_table_with_alias(table_part, table_aliases, used_aliases)
            on_condition = _rewrite_with_aliases(on_condition, table_aliases)
            join_kw = keyword if keyword != "JOIN" else "INNER JOIN"
            lines.append(f"{join_kw} {table_ref} ON {on_condition}")
        else:
            # Natural join without ON.
            table_ref = _parse_table_with_alias(content, table_aliases, used_aliases)
            lines.append(f"{keyword} {table_ref}")

    return lines


def _collect_aliases(clauses: List[Tuple[str, str]]) -> dict:
    """Scan FROM/JOIN clauses and return a table -> alias mapping."""
    table_aliases: dict = {}
    used_aliases: set = set()
    for keyword, content in clauses:
        if keyword == "FROM":
            if any(k in _JOIN_KEYWORDS for k, _ in clauses):
                _parse_table_with_alias(content, table_aliases, used_aliases)
        elif keyword in _JOIN_KEYWORDS:
            on_match = re.match(r"(.+?)\s+ON\s+(.+)", content, re.IGNORECASE)
            if on_match:
                _parse_table_with_alias(on_match.group(1).strip(), table_aliases, used_aliases)
            else:
                _parse_table_with_alias(content.strip(), table_aliases, used_aliases)
    return table_aliases


def _format_clauses(query: str) -> str:
    """Reassemble the query body with one clause per line."""
    clauses = _find_clauses(query)
    if not clauses or clauses[0][0] != "SELECT":
        raise SQLFormatError(f"Could not identify SELECT clause in query: {query}")

    table_aliases = _collect_aliases(clauses)

    lines: List[str] = ["SELECT"]
    select_content = _rewrite_with_aliases(clauses[0][1], table_aliases)
    lines.extend(_format_select_fields(select_content))

    from_lines = _format_from_and_joins(clauses, table_aliases)
    lines.extend(from_lines)

    for keyword, content in clauses[1:]:
        if keyword in _JOIN_KEYWORDS or keyword == "FROM":
            continue
        content = _rewrite_with_aliases(content, table_aliases)
        lines.append(f"{keyword} {content}")

    lines[-1] = _ensure_semicolon(lines[-1])
    return "\n".join(lines)


def _infer_description(query: str) -> str:
    """Create a generic description from query contents."""
    lowered = query.lower()
    tables = re.findall(r"FROM\s+(\w+)|JOIN\s+(\w+)", query, re.IGNORECASE)
    table_names = sorted({t for pair in tables for t in pair if t})
    table_phrase = ", ".join(table_names) if table_names else "the table"

    if "sum(" in lowered:
        return f"Returns the summed total from {table_phrase}."
    if "count(" in lowered:
        return f"Counts rows from {table_phrase}."
    if "avg(" in lowered or "average" in lowered:
        return f"Returns the average from {table_phrase}."
    if "join" in lowered:
        return f"Combines rows from {table_phrase}."
    if "where" in lowered:
        return f"Filters rows from {table_phrase}."
    if "group by" in lowered:
        return f"Summarizes rows from {table_phrase}."
    if "order by" in lowered:
        return f"Sorts rows from {table_phrase}."
    return f"Returns rows from {table_phrase}."


def _build_comment(description: str, created_by: str, create_date: str) -> str:
    return f"""/*
Created By: {created_by}
Create Date: {create_date}
Description: {description}
*/"""


def format_sql_query(
    query: str,
    description: str = "",
    created_by: str = "Walter Shields",
    create_date: Optional[str] = None,
) -> str:
    """
    Format a raw SQL query according to the course SQL standard.

    Returns a multi-line string containing a comment block followed by the
    formatted query ending with a semicolon.
    """
    if not query or not query.strip():
        raise SQLFormatError("Cannot format an empty query.")

    query = _normalize_whitespace(query)
    query = _uppercase_keywords(query)
    query = _ensure_semicolon(query)
    query_body = _format_clauses(query)

    if not description:
        description = _infer_description(query_body)
    if not create_date:
        create_date = _today()

    comment = _build_comment(description, created_by, create_date)
    return f"{comment}\n{query_body}"


# Transition phrases that usually follow a query inside a sentence.
_QUERY_BOUNDARY_RE = re.compile(
    r"""
    (SELECT\s+.+?)
    (?=
        \s+(?:into\s+the\s+execute\s+sql
           |in\s+the\s+execute\s+sql
           |and\s+(?:click|view|run|see|display|type|execute)
           |to\s+(?:view|see|display|run|execute)
           |then\s+(?:click|run|execute)
           |click\s+(?:the|run)
           |run\s+(?:the|it)
           |view\s+(?:the|results?)
           |see\s+the
           |showing
           |displaying
           |returning
           |giving
           |producing
           |followed
           |while
           |versus
           |vs
           |and\s+SELECT\b
           |query
           )
        \s+
      |\s*[.!?](?:\s|$)
      |$
    )
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def extract_first_query(text: str) -> Optional[str]:
    """Return the first SQL query found in `text`, or None."""
    match = _QUERY_BOUNDARY_RE.search(text)
    if not match:
        return None
    query = match.group(1).strip()
    query = _normalize_whitespace(query)
    query = query.rstrip(".!?;")
    return _ensure_semicolon(query)


def format_sql_in_text(text: str, created_by: str = "Walter Shields") -> str:
    """
    Find every inline SQL query in `text` and replace it with a formatted
    version that includes the standard comment block.

    Only queries that contain a FROM clause are reformatted; this avoids
    mangling prose such as "Run a SELECT * query" or "SELECT * returns 500 rows".
    """
    result = text
    for match in _QUERY_BOUNDARY_RE.finditer(text):
        raw = match.group(1).strip()
        if not re.search(r"\bFROM\b", raw, re.IGNORECASE):
            continue
        try:
            formatted = format_sql_query(raw, created_by=created_by)
        except SQLFormatError:
            continue
        result = result.replace(raw, formatted)
    return result
