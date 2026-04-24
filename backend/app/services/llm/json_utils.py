from __future__ import annotations

import json
import re


def extract_json_object(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty LLM response.")

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    json_text = cleaned[start:]
    try:
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(json_text)
        if not isinstance(payload, dict):
            raise ValueError("Top-level JSON value is not an object.")
        return payload
    except json.JSONDecodeError as exc:
        repaired = _repair_json_text(json_text)
        if repaired != json_text:
            try:
                decoder = json.JSONDecoder()
                payload, _ = decoder.raw_decode(repaired)
                if not isinstance(payload, dict):
                    raise ValueError("Top-level JSON value is not an object.")
                return payload
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to decode JSON from LLM response: {exc}") from exc


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
