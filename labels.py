"""Turns the plain-text names used in core.py into Markdown + LaTeX for the interface.

    tex("C_L,max landing flaps")  ->  "$C_{L,\\max}$ landing flaps"
    tex("rho (slug/ft^3)")        ->  "$\\rho$ (slug/ft³)"
"""
from __future__ import annotations

import re


def _sub(m):
    """Generic X_sub -> X_{sub}; descriptive (multi-letter) subscripts set upright."""
    base, sub = m.group(1), m.group(2)
    if len(sub) > 1 and not sub.isdigit():
        sub = r"\mathrm{" + sub + "}"
    return f"{base}_{{{sub}}}"


# (regex, LaTeX or callable) applied in order; longer / more specific patterns first
_SYMBOLS = [
    (r"K = 1/\(pi e AR\)", r"K = 1/(\pi e\,AR)"),
    (r"\(L/D\)max", r"(L/D)_{\max}"),
    (r"d\(eps\)/d\(alpha\)", r"d\varepsilon/d\alpha"),
    (r"V_C,min = k sqrt\(W/S\)", r"V_{C,\min} = k\sqrt{W/S}"),
    (r"V_app = 1\.3 V_s", r"V_{\mathrm{app}} = 1.3\,V_s"),
    (r"Delta C_D,0", r"\Delta C_{D,0}"),
    (r"dCL_max", r"\Delta C_{L,\max}"),
    (r"dCl_max", r"\Delta c_{l,\max}"),
    (r"C_L,max", r"C_{L,\max}"),
    (r"C_L,min", r"C_{L,\min}"),
    (r"V_C,min", r"V_{C,\min}"),
    (r"C_D,0", r"C_{D,0}"),
    (r"C_D,i", r"C_{D,i}"),
    (r"C_L\b", r"C_L"),
    (r"C_D\b", r"C_D"),
    (r"Cm_alpha,fus", r"C_{m_\alpha,\mathrm{fus}}"),
    (r"Cm_alpha", r"C_{m_\alpha}"),
    (r"Cm_ac", r"C_{m_{ac}}"),
    (r"Cm_0", r"C_{m_0}"),
    (r"Cm_de", r"C_{m_{\delta_e}}"),
    (r"Cm,cg", r"C_{m,cg}"),
    (r"\bCm\b", r"C_m"),
    (r"Ch_0", r"C_{h_0}"),
    (r"Ch_alpha", r"C_{h_\alpha}"),
    (r"Ch_delta", r"C_{h_\delta}"),
    (r"\bCh\b", r"C_h"),
    (r"alpha_0L", r"\alpha_{0L}"),
    (r"alpha_t", r"\alpha_t"),
    (r"\balpha\b", r"\alpha"),
    (r"eta_P", r"\eta_P"),
    (r"eta_h", r"\eta_h"),
    (r"\beta\(J\)", r"\eta(J)"),
    (r"\beta\b", r"\eta"),
    (r"\brho\b", r"\rho"),
    (r"mu_r", r"\mu_r"),
    (r"mu_b", r"\mu_b"),
    (r"mu_g", r"\mu_g"),
    (r"\bphi\b", r"\phi"),
    (r"\btau\b", r"\tau"),
    (r"\bsigma\b", r"\sigma"),
    (r"\|de\|", r"|\delta_e|"),
    (r"\bde\b", r"\delta_e"),
    (r"\bpi\b", r"\pi"),
    (r"\|n2\| / n1", r"|n_2|/n_1"),
    (r"\bn1\b", r"n_1"),
    (r"\bn2\b", r"n_2"),
    (r"\ba0\b", r"a_0"),
    (r"\|n2\| / n1", r"|n_2|/n_1"),
    (r"Ultimate \+ n", r"\text{Ultimate } +n"),
    (r"Ultimate - n", r"\text{Ultimate } -n"),
    (r"Governing \+ limit n", r"\text{Governing limit } +n"),
    (r"Governing - limit n", r"\text{Governing limit } -n"),
    (r"(?<![\w\\])([+-])n case", lambda m: m.group(1) + r"n\text{ case}"),
    (r"\|M\|", r"|M|"),
    (r"\|V\|", r"|V|"),
    (r"\bL/D\b", r"L/D"),
    (r"\bW/S\b", r"W/S"),
    (r"\bW/P\b", r"W/P"),
    (r"\bx/c\b", r"x/c"),
    (r"\bt/c\b", r"t/c"),
    (r"c'/c", r"c'/c"),
    (r"<=", r"\le"),
    (r">=", r"\ge"),
    (r"~(?= )", r"\approx"),
    (r"(?<=\d )x(?= )", r"\times"),
    (r"(?<![A-Za-z\\])([A-Za-z])_([A-Za-z0-9]+(?:,[A-Za-z0-9]+)*)", _sub),   # generic X_sub
    # standalone single-letter variables
    (r"(?<![\w\\'/|.\x04-])(V|M|J|n|w|y|e|a|c|h|k|W|S|b|K|q|T|D|L|P)(?![\w'/|.-])", lambda m: m.group(1)),
]
_COMPILED = [(re.compile(p), r) for p, r in _SYMBOLS]

_UNITS = [("lbf*in^2", "lbf·in²"), ("lbf*ft", "lbf·ft"), ("lbf*in", "lbf·in"), ("ft*lbf/s", "ft·lbf/s"),
          ("lb/(hp*h)", "lb/(hp·h)"), ("slug/ft^3", "slug/ft³"), ("ft/s^2", "ft/s²"), ("lbf/ft^2", "lbf/ft²"),
          ("ft^2", "ft²"), ("in^2", "in²"), ("in^4", "in⁴"), ("%MAC", "% MAC"),
          ("kg/(kW*h)", "kg/(kW·h)"), ("kg/m^3", "kg/m³"), ("N*m^2", "N·m²"), ("N*m", "N·m"),
          ("mm^2", "mm²"), ("mm^4", "mm⁴"), ("m^2", "m²")]


def tex(s) -> str:
    """Plain label -> Markdown with inline LaTeX. Non-strings are returned via str()."""
    if not isinstance(s, str):
        return str(s)
    for a, b in _UNITS:
        s = s.replace(a, b)
    s = s.replace(" — ", ": ").replace("—", ", ").replace("–", " to ")
    s = re.sub(r"(\d)\(([a-z])\)", "\\1\x04\\2\x05", s)      # protect legal refs like 23.2120(a)
    held: list[str] = []

    def hold(latex: str) -> str:
        held.append(latex.replace("|", r"\vert "))   # a bare | would split a markdown table cell
        return f"\x00{len(held) - 1}\x00"

    for rx, rep in _COMPILED:
        if callable(rep):
            s = rx.sub(lambda m, rep=rep: hold(rep(m)), s)
        else:
            s = rx.sub(lambda m, rep=rep: hold(rep), s)
    s = s.replace("*", r"\*").replace("_", r"\_")
    # math pieces separated only by spaces or simple operators are merged into one $...$
    s = re.sub(r"\x00(\d+)\x00", lambda m: "\x01" + held[int(m.group(1))] + "\x02", s)
    s = re.sub(r"\x02([ =/+]*)\x01", lambda m: m.group(1).replace(" ", r"\ "), s)
    return s.replace("\x01", "$").replace("\x02", "$").replace("\x04", "(").replace("\x05", ")")


def mpl(s: str) -> str:
    """Same conversion for matplotlib mathtext."""
    out = tex(s).replace(r"\*", "*").replace(r"\_", "_")
    return out.replace(r"\text{", r"\mathrm{")
