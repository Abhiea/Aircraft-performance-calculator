"""Display-unit conversion. The calculator core always works in imperial units; this module
converts labels and values for display (and converts SI inputs back to imperial).

Knots and nautical miles are kept in both systems, as is usual in aviation.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

IMPERIAL = "Imperial (ft, lbf, hp)"
SI = "SI (m, N, kW)"
SYSTEMS = [IMPERIAL, SI]

# imperial unit -> (SI unit, multiply imperial value by this to get SI)
_TO_SI = {
    "ft": ("m", 0.3048),
    "ft^2": ("m^2", 0.09290304),
    "in": ("mm", 25.4),
    "in^2": ("mm^2", 645.16),
    "in^4": ("mm^4", 416231.4256),
    "lbf": ("N", 4.4482216),
    "lbf/ft": ("N/m", 14.593903),
    "lbf/ft^2": ("Pa", 47.880259),
    "lbf/hp": ("N/kW", 5.965163),
    "lbf*ft": ("N*m", 1.3558179),
    "lbf*in": ("N*m", 0.11298483),
    "lbf*in^2": ("N*m^2", 0.0028696),
    "slug/ft^3": ("kg/m^3", 515.3788),
    "ft/s": ("m/s", 0.3048),
    "ft/min": ("m/s", 0.00508),
    "mph": ("km/h", 1.609344),
    "hp": ("kW", 0.7456999),
    "ft*lbf/s": ("kW", 0.0013558179),
    "lb/(hp*h)": ("kg/(kW*h)", 0.6082774),
    "lb/h": ("kg/h", 0.45359237),
    "psi": ("MPa", 0.0068947573),
}

_UNIT_RE = re.compile(r"([\(\[])([^()\[\]]*)([\)\]])")


def unit_factor(unit: str, system: str) -> tuple[str, float]:
    """(display unit, factor) for an imperial unit string."""
    if system == SI and unit in _TO_SI:
        return _TO_SI[unit]
    return unit, 1.0


def convert_label(label: str, system: str) -> tuple[str, float]:
    """Find the first '(unit)' or '[unit]' group with a known unit; swap it and return the factor."""
    if system != SI or not isinstance(label, str):
        return label, 1.0
    for m in _UNIT_RE.finditer(label):
        inner = m.group(2)
        head, sep, tail = inner.partition(",")
        u = head.strip()
        if u in _TO_SI:
            new_u, f = _TO_SI[u]
            new = f"{m.group(1)}{new_u}{sep}{tail}{m.group(3)}"
            return label[:m.start()] + new + label[m.end():], f
    return label, 1.0


def _scale(v, f):
    if f == 1.0 or isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)):
        return v
    return v * f


def convert_dict(d: dict, system: str) -> dict:
    out = {}
    for k, v in d.items():
        nk, f = convert_label(k, system)
        out[nk] = _scale(v, f)
    return out


def convert_columns(df: pd.DataFrame, system: str) -> pd.DataFrame:
    if system != SI:
        return df
    df = df.copy()
    ren = {}
    for c in df.columns:
        nc, f = convert_label(str(c), system)
        if f != 1.0 and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c] * f
        ren[c] = nc
    return df.rename(columns=ren)


def convert_rows(df: pd.DataFrame, system: str, label_col: str | None = None) -> pd.DataFrame:
    """Rows carry the unit (in the index or in label_col): scale numeric cells row by row."""
    if system != SI:
        return df
    df = df.copy().astype(object)
    labels = df.index if label_col is None else df[label_col]
    new_labels = []
    for idx, lab in zip(df.index, labels):
        nl, f = convert_label(str(lab), system)
        new_labels.append(nl)
        if f != 1.0:
            for c in df.columns:
                if c != label_col and "%" not in str(c):
                    df.at[idx, c] = _scale(df.at[idx, c], f)
    if label_col is None:
        df.index = new_labels
    else:
        df[label_col] = new_labels
    return df


_NUM_UNIT = re.compile(r"(\d[\d,]*\.?\d*) (ft|lbf|in)\b")


def convert_text(text: str, system: str) -> str:
    """Convert numbers written like '788 ft' inside free text."""
    if system != SI:
        return text

    def rep(m):
        v = float(m.group(1).replace(",", ""))
        u, f = _TO_SI[m.group(2)]
        return f"{v * f:,.0f} {u}" if v * f >= 100 else f"{v * f:.3g} {u}"
    return _NUM_UNIT.sub(rep, text)
