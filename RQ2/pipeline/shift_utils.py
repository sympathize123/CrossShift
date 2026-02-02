from __future__ import annotations

import re


def extract_shift_type_from_text(text: str) -> str:
    """Extract shift type from shift name."""
    tokens = str(text).strip().split()
    if len(tokens) >= 2 and tokens[-1].lower() == "shift":
        return tokens[-2]
    match = re.search(r"(\\w+)\\s+shift", str(text))
    return match.group(1) if match else str(text)


def parse_cat_and_shift_from_key(key: str):
    """Parse category and shift from combined key."""
    parts = key.split("_", 1)
    cat = parts[0]
    shift_full = parts[1].replace("_", " ") if len(parts) > 1 else parts[0]
    return cat, shift_full
