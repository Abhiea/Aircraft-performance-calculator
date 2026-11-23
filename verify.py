"""Equation checks.

Each equation on the Equations page is evaluated here a second time, written independently of core.py
(plain math, no calls to core's calculation functions), and compared with the value the calculator
reports. A few checks also compare against published reference data.

Run:  python verify.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import core

g, rho0, hp = 32.174, 0.0023769, 550.0
kt = 1.68781


def _rho(h):
    return rho0 * (1 - 6.87559e-6 * h) ** 4.25588


def _eta(i, J):
    if i.eta_mode == "constant":
        return i.eta_const
    xs, ys = np.asarray(i.eta_J, float), np.asarray(i.eta_vals, float)
    o = np.argsort(xs)
    return float(np.clip(np.interp(J, xs[o], ys[o], right=0.0), 0, 1))


def _pshaft(i, rho):
    return i.P_max * rho / rho0 * hp


def checks(inputs: core.Inputs | None = None, tol: float = 0.005) -> pd.DataFrame:
    r = core.run(inputs or core.Inputs())
    i = r.inputs
    rows = []

    def add(section, name, mine, calc, t=tol):
        ok = abs(mine - calc) <= t * max(abs(mine), 1e-12) if (mine == mine and calc == calc) else False
        rows.append({"Section": section, "Equation": name, "Independent value": mine,
                     "Calculator value": calc, "Result": "Pass" if ok else "Fail"})

    W, S, b = i.W, i.S, i.b
    AR = b * b / S
    K = 1 / (math.pi * i.e * AR)
    rho = _rho(i.altitude)
    c = S / b

    # atmosphere (also against the published standard-atmosphere value at 10,000 ft)
    add("Atmosphere", "Density at 10,000 ft vs standard table (0.0017556)", 0.0017556, float(core.density(10000.0)), 0.001)
    add("Atmosphere", "Density at the chosen altitude", rho, r.summary["Density at altitude (slug/ft^3)"])

    # geometry and aerodynamics
    add("Geometry", "Aspect ratio AR = b^2/S", AR, r.summary["Aspect ratio AR"])
    add("Geometry", "Induced drag factor K = 1/(pi e AR)", K, r.summary["K = 1/(pi e AR)"])
    a_w = math.pi * AR / (1 + math.sqrt(1 + (math.pi * AR / i.a0_wing) ** 2))
    add("Aerodynamics", "Wing lift slope (Helmbold)", a_w, r.stab["Wing 3-D lift slope a_w (1/rad)"])
    add("Aerodynamics", "(L/D)max = 1/sqrt(4 C_D0 K)", 1 / math.sqrt(4 * i.CD0 * K), r.summary["(L/D)max"])
    Vs = math.sqrt(2 * W / (rho * S * i.CLmax_clean))
    add("Aerodynamics", "Stall speed V_s = sqrt(2W/(rho S C_Lmax))", Vs / kt, r.summary["V_stall clean (kt)"])

    # level flight at cruise speed
    V = i.V_cruise
    q = 0.5 * rho * V * V
    CL = W / (q * S)
    D = q * S * (i.CD0 + K * CL * CL)
    pt = core.power_table(i, rho=rho, V=np.array([V]))
    add("Power", "Power required P_R = D V at cruise", D * V / hp, float(pt["P_R (hp)"].iloc[0]))
    J = V / (i.rpm / 60 * i.D_prop / 12)
    PA = _eta(i, J) * _pshaft(i, rho)
    add("Power", "Power available P_A = eta P at cruise", PA / hp, float(pt["P_A (hp)"].iloc[0]))
    add("Power", "Rate of climb (P_A - P_R)/W at cruise", (PA - D * V) / W * 60, float(pt["R/C (ft/min)"].iloc[0]))

    # maximum level speed by bisection on P_A = P_R
    def excess(Vx):
        qx = 0.5 * rho * Vx * Vx
        CLx = W / (qx * S)
        return _eta(i, Vx / (i.rpm / 60 * i.D_prop / 12)) * _pshaft(i, rho) - qx * S * (i.CD0 + K * CLx * CLx) * Vx
    lo, hi = 1.5 * Vs, i.V_hi
    if excess(lo) > 0 > excess(hi):
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
        add("Power", "Maximum level speed (P_A = P_R)", lo / kt, r.summary["V_max level (kt)"], 0.01)

    # Part 23 takeoff climb gradient
    CD0_to = i.CD0 + i.dCD0_flaps_TO
    Vs_to = math.sqrt(2 * W / (rho0 * S * i.CLmax_TO))
    best = -1
    for Vx in np.linspace(1.1 * Vs_to, 3.5 * Vs_to, 2000):
        qx = 0.5 * rho0 * Vx * Vx
        T = _eta(i, Vx / (i.rpm / 60 * i.D_prop / 12)) * _pshaft(i, rho0) / Vx
        best = max(best, (T - qx * S * (CD0_to + K * (W / (qx * S)) ** 2)) / W)
    add("Climb", "Takeoff climb gradient (T - D)/W", 100 * best, r.summary["Takeoff climb gradient (%)"], 0.01)

    # takeoff ground roll (Anderson)
    phi = (16 * i.h_wing / b) ** 2 / (1 + (16 * i.h_wing / b) ** 2)
    Vs_f = math.sqrt(2 * W / (rho * S * i.CLmax_TO))
    V7 = 0.7 * i.VLO_factor * Vs_f
    CLg = min(math.pi * i.e * AR * i.mu_roll / (2 * phi), i.CLmax_TO)
    q7 = 0.5 * rho * V7 * V7
    T7 = _eta(i, V7 / (i.rpm / 60 * i.D_prop / 12)) * _pshaft(i, rho) / V7
    F = T7 - q7 * S * (CD0_to + phi * K * CLg ** 2) - i.mu_roll * (W - q7 * S * CLg)
    add("Takeoff and landing", "Takeoff ground roll 1.44 W^2/(g rho S C_Lmax F)",
        1.44 * W * W / (g * rho * S * i.CLmax_TO * F), r.summary["Takeoff roll, flaps (ft)"])

    # landing ground roll (Anderson)
    Vs_l = math.sqrt(2 * W / (rho * S * i.CLmax_land))
    V7 = 0.7 * i.VTD_factor * Vs_l
    q7 = 0.5 * rho * V7 * V7
    Fl = q7 * S * (i.CD0 + i.dCD0_flaps + phi * K * i.CL_roll_land ** 2) + i.mu_brake * (W - q7 * S * i.CL_roll_land)
    add("Takeoff and landing", "Landing ground roll 1.69 W^2/(g rho S C_Lmax F)",
        1.69 * W * W / (g * rho * S * i.CLmax_land * Fl), r.summary["Landing roll, flaps (ft)"])

    # Breguet range
    Wf = r.weights["Full fuel (lbf)"]
    cc = i.SFC / (hp * 3600)
    V_md = math.sqrt(2 * W / (rho * S)) * (K / i.CD0) ** 0.25
    R = _eta(i, V_md / (i.rpm / 60 * i.D_prop / 12)) / cc / math.sqrt(4 * i.CD0 * K) * math.log(W / (W - Wf))
    add("Cruise", "Breguet maximum range", R / 6076.12, r.cruise["Max range (nmi)"])
    V_mp = math.sqrt(2 * W / (rho * S)) * (K / (3 * i.CD0)) ** 0.25
    E = (_eta(i, V_mp / (i.rpm / 60 * i.D_prop / 12)) / cc * 0.25 * (3 / (K * i.CD0 ** (1 / 3))) ** 0.75
         * math.sqrt(2 * rho * S) * ((W - Wf) ** -0.5 - W ** -0.5))
    add("Cruise", "Breguet maximum endurance", E / 3600, r.cruise["Max endurance (h)"])

    # glide
    add("Glide", "Minimum glide angle atan(1/(L/D)max)",
        math.degrees(math.atan(math.sqrt(4 * i.CD0 * K))), r.glide_summary["clean"]["Min glide angle (deg)"])

    # weights
    items = pd.DataFrame(i.weight_items)
    e_ = items[items["Type"] == "empty"]
    We = e_["Weight (lbf)"].sum()
    add("Weights", "Empty weight sum(W_i)", We, r.weights["Empty weight (lbf)"])
    add("Weights", "Empty CG sum(W_i x_i)/sum(W_i)", (e_["Weight (lbf)"] * e_["Arm (in)"]).sum() / We, r.weights["Empty CG (in)"])

    # stability
    ARh = i.b_h ** 2 / i.S_h
    a_t = math.pi * ARh / (1 + math.sqrt(1 + (math.pi * ARh / i.a0_tail) ** 2))
    deps = i.deps_override if i.deps_override > 0 else 2 * a_w / (math.pi * AR)
    VH = i.S_h * i.l_h / (S * c)
    Cmf = i.Kf * i.w_fus ** 2 * i.L_fus / (c * S) * 180 / math.pi
    hn = i.h_ac - Cmf / a_w + i.eta_h * VH * a_t / a_w * (1 - deps)
    add("Stability", "Tail volume V_H = S_h l_h/(S c)", VH, r.stab["Horizontal tail volume V_H"])
    add("Stability", "Neutral point h_n", 100 * hn, r.stab["Neutral point (%MAC)"])
    add("Stability", "Static margin at aft CG", 100 * hn - r.weights["Aft CG (%MAC)"], r.stab["Static margin, aft CG (%MAC)"])

    # V-n
    WS = W / S
    n1 = min(2.1 + 24000 / (W + 10000), i.n1_cap)
    Vs1 = math.sqrt(2 * W / (rho0 * S * i.CLmax_clean))
    add("V-n", "Limit load factor n_1", n1, r.vn["n1 limit"])
    add("V-n", "Manoeuvring speed V_A = V_S1 sqrt(n_1)", Vs1 * math.sqrt(n1) / kt, r.vn["V_A (kt)"])
    VC = i.VC_coef * math.sqrt(WS)
    add("V-n", "Design cruise speed V_C = 33 sqrt(W/S)", VC, r.vn["V_C (kt)"])
    a_g = a_w
    mu = 2 * WS / (rho0 * c * a_g * g)
    Kg = 0.88 * mu / (5.3 + mu)
    add("V-n", "Gust load factor at V_C", 1 + Kg * rho0 * i.Ude_C * VC * kt * a_g / (2 * WS), r.vn["n gust + at V_C"])

    # spar: Schrenk root moment has a closed form for the cantilever wing
    s = b / 2
    L_half = r.vn["Ultimate + n"] * (W - i.W_wing_frac * W) / 2
    M_root = L_half * 0.5 * (s / 2 + 4 * s / (3 * math.pi))
    add("Spar", "Cantilever root moment (Schrenk, closed form)", M_root, r.spar_summary["Root moment, cantilever (lbf*ft)"], 0.01)
    if i.strut:
        add("Spar", "Strut reaction R_z = M(0)/y_s", r.spar_summary["Root moment, cantilever (lbf*ft)"] / i.y_strut,
            r.spar_summary["Strut vertical reaction R_z (lbf)"])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = checks()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\n{(df['Result'] == 'Pass').sum()} of {len(df)} checks pass")
