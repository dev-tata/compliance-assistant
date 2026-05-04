from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty LLM response.")

    cleaned = _strip_code_fence(raw_text.strip())
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    json_text = cleaned[start:]
    parse_debug: dict[str, Any] = {
        "response_length": len(raw_text),
        "attempted_repair": False,
        "fallback_used": False,
        "steps": [],
    }

    try:
        return _parse_json_text(json_text)
    except json.JSONDecodeError as exc:
        parse_debug["steps"].append(_format_json_error(exc, stage="initial"))
    except ValueError as exc:
        if hasattr(exc, "parse_debug"):
            parse_debug.update(exc.parse_debug)
        raise

    fallback_text = _extract_first_balanced_json_object(json_text)
    if fallback_text != json_text:
        parse_debug["fallback_used"] = True
        parse_debug["steps"].append("fallback to first balanced JSON object")
        try:
            return _parse_json_text(fallback_text)
        except json.JSONDecodeError as exc:
            parse_debug["steps"].append(_format_json_error(exc, stage="fallback"))
        json_text = fallback_text

    repaired = _normalize_json_string_content(json_text)
    if repaired != json_text:
        parse_debug["attempted_repair"] = True
        parse_debug["steps"].append("normalized string content")
        try:
            return _parse_json_text(repaired)
        except json.JSONDecodeError as exc:
            parse_debug["steps"].append(_format_json_error(exc, stage="normalize_strings"))
        json_text = repaired

    repaired = _repair_json_text(json_text)
    if repaired != json_text:
        parse_debug["attempted_repair"] = True
        parse_debug["steps"].append("repaired common JSON issues")
        try:
            return _parse_json_text(repaired)
        except json.JSONDecodeError as exc:
            parse_debug["steps"].append(_format_json_error(exc, stage="repair_text"))
        json_text = repaired

    appended = _append_missing_closing_tokens(json_text)
    if appended != json_text:
        parse_debug["attempted_repair"] = True
        parse_debug["steps"].append("appended missing closing tokens")
        try:
            return _parse_json_text(appended)
        except json.JSONDecodeError as exc:
            parse_debug["steps"].append(_format_json_error(exc, stage="pad_closing_tokens"))

    exc = ValueError(
        "Failed to decode JSON from LLM response: could not recover a top-level JSON object."
    )
    exc.parse_debug = parse_debug
    raise exc


def _parse_json_text(json_text: str) -> dict:
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(json_text)
    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON value is not an object.")
    return payload


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    stack: list[str] = []
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            if in_string:
                escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    return text[start : index + 1]
    return text[start:]


def _normalize_json_string_content(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    i = 0

    while i < len(text):
        char = text[i]

        if escape:
            result.append(char)
            escape = False
            i += 1
            continue

        if char == "\\" and in_string:
            next_char = text[i + 1] if i + 1 < len(text) else ""
            if next_char not in '"\\/bfnrtu':
                result.append('\\\\')
                i += 1
                continue
            result.append(char)
            escape = True
            i += 1
            continue

        if char == '"':
            if in_string:
                next_non_ws = _next_non_whitespace_char(text, i + 1)
                if next_non_ws is not None and next_non_ws not in ':,}]':
                    result.append('\\"')
                    i += 1
                    continue
            result.append(char)
            in_string = not in_string
            i += 1
            continue

        if in_string and char in '\n\r\t':
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            else:
                result.append('\\t')
            i += 1
            continue

        result.append(char)
        i += 1

    return "".join(result)


def _next_non_whitespace_char(text: str, start: int) -> str | None:
    index = start
    while index < len(text):
        if not text[index].isspace():
            return text[index]
        index += 1
    return None


def _append_missing_closing_tokens(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            stack.append('{')
        elif char == '[':
            stack.append('[')
        elif char == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif char == ']' and stack and stack[-1] == '[':
            stack.pop()

    if not stack:
        return text

    closing = ''.join('}' if item == '{' else ']' for item in reversed(stack))
    return text + closing


def _repair_json_text(json_text: str) -> str:
    repaired = json_text
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
    repaired = _insert_missing_colons_between_key_value_tokens(repaired)
    return repaired


def _insert_missing_colons_between_key_value_tokens(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    i = 0

    while i < len(text):
        char = text[i]
        result.append(char)

        if escape:
            escape = False
            i += 1
            continue

        if char == "\\" and in_string:
            escape = True
            i += 1
            continue

        if char == '"':
            in_string = not in_string
            if not in_string:
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == '"':
                    result.append(":")
            i += 1
            continue

        i += 1

    return "".join(result)


def _format_json_error(exc: json.JSONDecodeError | str, *, stage: str | None = None) -> dict[str, Any]:
    if isinstance(exc, str):
        message = exc
        lineno = None
        colno = None
        pos = None
    else:
        message = str(exc)
        lineno = exc.lineno
        colno = exc.colno
        pos = exc.pos
    payload = {
        "stage": stage,
        "message": message,
        "line": lineno,
        "column": colno,
        "pos": pos,
    }
    return {k: v for k, v in payload.items() if v is not None}
