from __future__ import annotations


def strtobool(val: str) -> int:
    """
    Minimal drop-in for distutils.util.strtobool (removed in Python 3.12).
    Returns 1 for truthy strings and 0 for falsy strings.
    """
    v = val.strip().lower()
    if v in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if v in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")

