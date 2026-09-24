"""Canonicalize asteroid designations so different catalogs join correctly.

An asteroid can be identified several ways in this study's inputs:
  * a permanent NUMBER, written plainly ("5") or, for numbers >= 100000, in MPC
    PACKED form with a leading letter ("A0001" = 100001; "z9999" = 619999;
    "~AZaz" base-62 for >= 620000);
  * a PROVISIONAL designation, written human-readable ("2013 PL32", "2013PL32")
    or MPC PACKED ("K13P32L" = 2013 PL32).
A NAMED asteroid is always also numbered and appears by its number in these
catalogs, so no name handling is required.

The AFP family lists use plain numbers + human provisional designations; the
NEOWISE diameter catalog (neowise_mainbelt.csv) uses MPC packed forms. Matching
them therefore requires reducing any form to a single CANONICAL key:
  * an ``int`` for a numbered object, or
  * a normalized provisional string ``"<year><half-month><letter><cycle>"``
    (e.g. ``"2013PL32"``; cycle omitted when zero).

``canonical()`` accepts any of the forms above and returns that key (or ``None``
if it cannot be parsed). Reference: MPC "Packed Provisional and Permanent
Designations" (minorplanetcenter.net/iau/info/PackedDes.html).
"""

from __future__ import annotations

import re

# base-62 alphabet: index == value (0-9, A-Z = 10-35, a-z = 36-61)
_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_VAL = {c: i for i, c in enumerate(_B62)}
_CENTURY = {"I": 18, "J": 19, "K": 20}                     # provisional century letter
_HUMAN_PROV = re.compile(r"^(\d{4})\s*([A-Za-z]{1,2})(\d*)$")


def unpack_number(s: str):
    """MPC packed-or-plain permanent designation -> int, else None."""
    s = s.strip()
    if s.isdigit():                                        # plain "00005" / "12345"
        return int(s)
    if len(s) == 5 and s[0] in _VAL and s[1:].isdigit():   # "A0001" -> 100001
        return _VAL[s[0]] * 10000 + int(s[1:])
    if s.startswith("~") and len(s) == 5:                  # "~0000" base-62, >= 620000
        v = 0
        for c in s[1:]:
            if c not in _VAL:
                return None
            v = v * 62 + _VAL[c]
        return 620000 + v
    return None


def unpack_provisional(s: str):
    """MPC packed provisional (7 chars, e.g. 'K13P32L') -> '2013PL32', else None."""
    s = s.strip()
    if len(s) != 7 or s[0] not in _CENTURY or not s[1:3].isdigit():
        return None
    if s[4] not in _VAL or not s[5].isdigit():
        return None
    year = _CENTURY[s[0]] * 100 + int(s[1:3])
    cycle = _VAL[s[4]] * 10 + int(s[5])                    # packed cycle count
    half, second = s[3], s[6]                              # half-month + order letters
    cyc = "" if cycle == 0 else str(cycle)
    return f"{year}{half}{second}{cyc}"


def canonical(s):
    """Reduce any designation form to a canonical key: int, provisional str, or None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    n = unpack_number(s)
    if n is not None:
        return n
    p = unpack_provisional(s)
    if p is not None:
        return p
    m = _HUMAN_PROV.match(s.replace(" ", ""))              # human "2013PL32" / "1994 VX3"
    if m:
        year, letters, cyc = m.groups()
        cyc = str(int(cyc)) if cyc and int(cyc) != 0 else ""
        return f"{year}{letters.upper()}{cyc}"
    return None
