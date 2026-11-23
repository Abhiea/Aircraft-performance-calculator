"""Report-ready matplotlib figures (titles, axis labels with units, curve labels), math set with mathtext."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import KT, Results, density, power_table, K_ind
import units as U

_SYS = U.IMPERIAL
_MT = {"ft^2": r"ft$^2$", "m^2": r"m$^2$", "lbf*ft": r"lbf$\cdot$ft", "N*m": r"N$\cdot$m"}


def set_units(system: str):
    global _SYS
    _SYS = system


def _c(x, unit):
    """Convert imperial values to the display system."""
    return np.asarray(x) * U.unit_factor(unit, _SYS)[1]


def _u(unit):
    """Display unit string for an imperial unit."""
    u = U.unit_factor(unit, _SYS)[0]
    return _MT.get(u, u)

plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 10})


def _fig(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=110)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def _legend2(ax, ax2, **kw):
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, **kw)


def power_curves(r: Results, altitudes=None):
    i = r.inputs
    altitudes = altitudes or [i.altitude]
    fig, ax = _fig("Power required and available vs airspeed", r"Airspeed $V$ (kt)", f"Power ({_u('hp')})")
    for k, h in enumerate(altitudes):
        df = power_table(i, rho=float(density(h)))
        df = df[~df["Stalled"]]
        hl = f"{float(_c(h, 'ft')):,.0f} {_u('ft')}"
        ax.plot(df["V (kt)"], _c(df["P_R (hp)"], "hp"), f"C{k}", label=rf"$P_R$, {hl}")
        ax.plot(df["V (kt)"], _c(df["P_A (hp)"], "hp"), f"C{k}--", label=rf"$P_A$, {hl}")
    ax.set_ylim(0, float(_c(i.P_max, "hp")) * 1.15)
    ax.legend()
    return fig


def rate_of_climb(r: Results):
    df = r.power[~r.power["Stalled"]]
    fig, ax = _fig(f"Rate of climb vs airspeed ({float(_c(r.inputs.altitude, 'ft')):,.0f} {_u('ft')})",
                   r"Airspeed $V$ (kt)", f"$R/C$ ({_u('ft/min')})")
    ax.plot(df["V (kt)"], _c(df["R/C (ft/min)"], "ft/min"), label=r"Max $R/C$ at full power")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    return fig


def climb_vs_altitude(r: Results):
    fig, ax = _fig("Maximum rate of climb and time to climb vs altitude", f"Max $R/C$ ({_u('ft/min')})", f"Altitude $h$ ({_u('ft')})")
    alt = _c(r.climb["Altitude (ft)"], "ft")
    ax.plot(_c(r.climb["Max R/C (ft/min)"], "ft/min"), alt, label=r"Max $R/C$")
    ax.axvline(float(_c(100, "ft/min")), color="C3", ls=":", label=f"Service ceiling rate ({float(_c(100, 'ft/min')):.3g} {_u('ft/min')})")
    ax2 = ax.twiny()
    ax2.plot(r.climb["Time to climb (min)"], alt, "C1--", label="Time to climb")
    ax2.set_xlabel("Time to climb (min)")
    _legend2(ax, ax2, loc="upper right")
    ax.set_xlim(left=0)
    return fig


def drag_polar(r: Results):
    i = r.inputs
    K = K_ind(i)
    CL = np.linspace(-0.3, max(i.CLmax_clean, i.CLmax_land), 200)
    fig, ax = _fig("Drag polar", r"Drag coefficient $C_D$", r"Lift coefficient $C_L$")
    ax.plot(i.CD0 + K * CL ** 2, CL, label=rf"Clean: $C_D = {i.CD0:.4f} + {K:.4f}\,C_L^2$")
    ax.plot(i.CD0 + i.dCD0_flaps + K * CL ** 2, CL, "--", label=r"Landing flaps ($\Delta C_{D,0}$ added)")
    CLs = np.sqrt(i.CD0 / K)
    ax.plot([0, 2 * i.CD0], [0, CLs], "k:", lw=0.8)
    ax.plot(2 * i.CD0, CLs, "ko", label=rf"$(L/D)_{{\max}} = {r.cruise['(L/D)max']:.2f}$")
    ax.set_xlim(left=0)
    ax.legend()
    return fig


def range_vs_power(r: Results):
    df = r.power_settings.dropna()
    fig, ax = _fig("Range and endurance vs power setting", "Power setting (% of maximum)", "Range (nmi)")
    ax.plot(df["Power setting (%)"], df["Range (nmi)"], "o-", label="Range")
    ax2 = ax.twinx()
    ax2.plot(df["Power setting (%)"], df["Endurance (h)"], "s--C1", label="Endurance")
    ax2.set_ylabel("Endurance (h)")
    _legend2(ax, ax2)
    return fig


def turning(r: Results):
    df = r.turn.dropna(subset=["Turn radius (ft)"])
    fig, ax = _fig("Level turn performance", r"Airspeed $V$ (kt)", f"Turn radius $R$ ({_u('ft')})")
    ax.plot(df["V (kt)"], _c(df["Turn radius (ft)"], "ft"), label="Minimum turn radius")
    ax2 = ax.twinx()
    ax2.plot(df["V (kt)"], df["Turn rate (deg/s)"], "C1--", label=r"Maximum turn rate $\omega$")
    ax2.set_ylabel(r"Turn rate $\omega$ (deg/s)")
    _legend2(ax, ax2)
    return fig


def hodograph(r: Results):
    fig, ax = _fig("Glide hodograph", r"Horizontal speed $V$ (kt)", f"Sink rate ({_u('ft/min')})")
    for cfg, g in r.glide.groupby("Config"):
        ax.plot(g["V (kt)"], _c(g["Sink rate (ft/min)"], "ft/min"), label={"clean": "Clean", "flaps": "Landing flaps"}.get(cfg, cfg))
    ax.invert_yaxis()
    ax.legend()
    return fig


def vn(r: Results):
    v, c = r.vn, r.vn_curves
    fig, ax = _fig("V-n diagram (sea level, maximum takeoff weight)", r"Equivalent airspeed $V_E$ (kt)", r"Load factor $n$")
    V = c["V"] / KT
    ax.plot(V, c["pos"], "C0", label="Manoeuvre envelope")
    ax.plot(V, c["neg"], "C0")
    VD = v["V_D (kt)"]
    ax.plot([VD, VD], [c["neg"][-1], c["pos"][-1]], "C0")
    for key, col, Ug in [("gust_C", "C1", r.inputs.Ude_C), ("gust_D", "C2", r.inputs.Ude_D)]:
        Vg, up, dn = c[key]
        ax.plot(Vg / KT, up, col + "--", label=rf"Gust $U_{{de}}$ = {float(_c(Ug, 'ft/s')):.3g} {_u('ft/s')}")
        ax.plot(Vg / KT, dn, col + "--")
    top = v["Governing + limit n"] + 0.3
    for name, sub in [("V_A", "A"), ("V_C", "C"), ("V_D", "D")]:
        ax.axvline(v[f"{name} (kt)"], color="grey", ls=":", lw=0.8)
        ax.text(v[f"{name} (kt)"], top, rf"$V_{sub}$", ha="center")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(v["Governing - limit n"] - 0.8, v["Governing + limit n"] + 0.8)
    ax.set_xlim(0, VD * 1.05)
    ax.legend(loc="lower left")
    return fig


def spar_loads(r: Results):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 8.5), dpi=110, sharex=True)
    cases = [(r.spar, rf"$+n_{{ult}} = {r.vn['Ultimate + n']:.2f}$", "-"),
             (r.spar_neg, rf"$-n_{{ult}} = {abs(r.vn['Ultimate - n']):.2f}$", ":")]
    for df, lab, ls in cases:
        y = _c(df["y (ft)"], "ft")
        a1.plot(y, _c(df["w (lbf/ft)"], "lbf/ft"), "C0" + ls, label=lab)
        a2.plot(y, _c(df["V cantilever (lbf)"], "lbf"), "C0" + ls, label=f"Cantilever, {lab}")
        a3.plot(y, _c(df["M cantilever (lbf*ft)"], "lbf*ft"), "C0" + ls, label=f"Cantilever, {lab}")
        if "V strut-braced (lbf)" in df:
            st = "C1" + ("--" if ls == "-" else ":")
            a2.plot(y, _c(df["V strut-braced (lbf)"], "lbf"), st, label=f"Strut-braced, {lab}")
            a3.plot(y, _c(df["M strut-braced (lbf*ft)"], "lbf*ft"), st, label=f"Strut-braced, {lab}")
    a1.set_title("Half-wing loads at ultimate load factors (Schrenk)")
    a1.set_ylabel(f"$w$ ({_u('lbf/ft')})"); a2.set_ylabel(f"Shear $V$ ({_u('lbf')})"); a3.set_ylabel(f"Moment $M$ ({_u('lbf*ft')})")
    a3.set_xlabel(f"Spanwise position $y$ from root ({_u('ft')})")
    for a in (a1, a2, a3):
        a.grid(True, alpha=0.3); a.axhline(0, color="k", lw=0.5); a.legend(fontsize=7)
    fig.tight_layout()
    return fig


def propulsion(r: Results):
    df = r.propulsion
    fig, ax = _fig("Engine and propeller combination: thrust and power available", r"Airspeed $V$ (kt)", f"Thrust $T$ ({_u('lbf')})")
    ax2 = ax.twinx(); ax2.set_ylabel(f"Power available $P_A$ ({_u('hp')})")
    for k, (h, hv) in enumerate([("0", 0.0), ("5,000", 5000.0), ("10,000", 10000.0)]):
        hl = f"{float(_c(hv, 'ft')):,.0f} {_u('ft')}"
        ax.plot(df["V (kt)"], _c(df[f"Thrust {h} ft (lbf)"], "lbf"), f"C{k}-", label=rf"$T$, {hl}")
        ax2.plot(df["V (kt)"], _c(df[f"P_A {h} ft (hp)"], "hp"), f"C{k}--", label=rf"$P_A$, {hl}")
    _legend2(ax, ax2, fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
    return fig


def cg_envelope(r: Results):
    df, w, s = r.loading_cases, r.weights, r.stab
    fig, ax = _fig("Loading cases: weight vs centre of gravity", r"CG position (% $\bar{c}$)", f"Weight $W$ ({_u('lbf')})")
    ax.scatter(df["CG (%MAC)"], _c(df["Weight (lbf)"], "lbf"), s=18, label="Loading combinations")
    ax.axhline(float(_c(w["MTOW (lbf)"], "lbf")), color="k", ls="--", lw=0.8, label="MTOW")
    ax.axvline(s["Neutral point (%MAC)"], color="C3", label=rf"Neutral point $h_n$ = {s['Neutral point (%MAC)']:.1f}%")
    ax.axvline(s["Aft CG limit for min SM (%MAC)"], color="C3", ls=":", label="Aft limit for minimum static margin")
    ax.legend(fontsize=8)
    return fig


def cm_alpha(r: Results):
    df = r.cm_alpha
    fig, ax = _fig(r"Pitching moment about the CG vs angle of attack ($\delta_e = 0$)",
                   r"Wing angle of attack $\alpha$ (deg)", r"$C_{m,cg}$")
    ax.plot(df["alpha (deg)"], df["Cm fwd CG"], label=f"Forward CG ({r.weights['Forward CG (%MAC)']:.1f}% MAC)")
    ax.plot(df["alpha (deg)"], df["Cm aft CG"], "--", label=f"Aft CG ({r.weights['Aft CG (%MAC)']:.1f}% MAC)")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    return fig


def hinge(r: Results):
    df = r.hinge
    fig, ax = _fig("Elevator hinge moment coefficient", r"Wing angle of attack $\alpha$ (deg)", r"$C_h$")
    for de, g in df.groupby("de (deg)"):
        ax.plot(g["alpha (deg)"], g["Ch"], label=rf"$\delta_e = {de:+.0f}^\circ$")
    ax.legend(fontsize=8, ncol=2)
    return fig


ALL = {"power_curves": power_curves, "rate_of_climb": rate_of_climb, "climb_vs_altitude": climb_vs_altitude,
       "drag_polar": drag_polar, "range_vs_power": range_vs_power, "turning": turning,
       "glide_hodograph": hodograph, "vn_diagram": vn, "spar_loads": spar_loads,
       "propulsion": propulsion, "cg_envelope": cg_envelope, "cm_alpha": cm_alpha, "hinge_moment": hinge}


def save_all(r: Results, folder: str = "figures"):
    os.makedirs(folder, exist_ok=True)
    for name, fn in ALL.items():
        f = fn(r)
        f.savefig(os.path.join(folder, f"{name}.png"), bbox_inches="tight")
        plt.close(f)
    return folder
