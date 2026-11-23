"""The 'How to use' and 'Equations' pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import core
import labels
import units

tex = labels.tex
BADGE = {"estimate": ":gray-badge[Estimate]", "placeholder": ":orange-badge[Placeholder]"}
XFLR5_BADGE = ":blue-badge[XFLR5]"


def badges(meta: dict) -> str:
    out = BADGE.get(meta.get("status"), "")
    if meta.get("xflr5"):
        out += (" " if out else "") + XFLR5_BADGE
    return out


# ================================================================== How to use
GROUP_HELP = {
    "Weights & CG": "Maximum takeoff weight and the component weight table. "
                    "Each row of the table is an item with its weight, its distance aft of the datum and a type. "
                    "The table sets the empty weight, the useful load and the CG range.",
    "Wing geometry": "Span, area and the wing's height above the ground (used for ground effect on the runway).",
    "Aerodynamics": "The drag polar ($C_{D,0}$ and $e$), maximum and minimum lift coefficients, and airfoil data "
                    "for the wing. These drive almost every result.",
    "High-lift (flaps)": "Maximum lift coefficients and extra drag with takeoff and landing flaps.",
    "Propulsion": "Engine power and speed, propeller diameter, the propeller efficiency curve and fuel consumption.",
    "Tail & stability": "Tail sizes and positions, airfoil data, fuselage size, and the elevator values used for "
                        "trim and hinge moments.",
    "Flight condition": "The altitude and cruise speed for the results, and the speed range of the tables.",
    "Takeoff & landing": "Runway friction and the liftoff and touchdown speeds.",
    "Loads (V-n)": "The rules that set the load factors and design speeds: limit load factors, gust velocities, "
                   "speed ratios and the factor of safety.",
    "Spar": "Strut layout, spar positions and depths, and the material allowables used to size the spar.",
    "Requirements": "Pass or fail limits used on the Overview tab: runway length, Mach number, climb gradients "
                    "and minimum static margin.",
}


def how_to():
    st.title("How to use")

    st.markdown(
        "This app estimates how a small propeller aircraft performs: speeds, climb, takeoff and landing distances, "
        "range, stability and wing loads. You enter the aircraft's numbers on the left, and the results update "
        "straight away.")

    st.markdown("#### Quick start")
    st.markdown(
        "1. The app opens with a Cessna 172S already filled in.\n"
        "2. Choose **Imperial** or **SI** units at the top left.\n"
        "3. Open a group on the left (for example *Wing geometry*) and change a number.\n"
        "4. Look at the **Overview** tab to see what passes and what fails.\n"
        "5. Click **Save inputs** to keep your design. Click **Load inputs** to open it again.")

    st.markdown("#### The tabs")
    st.markdown(md(pd.DataFrame([
        ("Overview", "Pass or fail checks and the key results"),
        ("Performance", "Speed, climb, takeoff and landing, range, turns and gliding"),
        ("Weight and balance", "Empty weight, payload and centre of gravity"),
        ("Stability and control", "How stable the aircraft is in pitch and how much elevator it needs"),
        ("Loads and structure", "The V-n diagram and the loads on the wing spar"),
        ("Export", "Download tables, plots and your inputs"),
    ], columns=["Tab", "What it shows"])))

    st.markdown("#### Useful buttons")
    st.markdown(
        "- **Pin as baseline** remembers the current design. After that, each change shows how much better or "
        "worse the results got.\n"
        "- **Reset to Cessna 172 values** puts every input back to the start.\n"
        "- The **?** next to an input explains what it is and where the value came from.")

    st.markdown("#### Labels on inputs")
    st.markdown(
        f"- {BADGE['placeholder']} a stand-in value. Replace it with real data.\n"
        f"- {BADGE['estimate']} a rough guess. Fine to start with.\n"
        f"- {XFLR5_BADGE} for your own aircraft, get this value from XFLR5.")

    st.markdown("#### Good to know")
    st.markdown(
        "- The app does not work out drag or lift from the aircraft's shape. Those numbers are inputs.\n"
        "- Results assume a standard day and speeds below about Mach 0.3.\n"
        "- It is for design and learning, not certification.")

    st.markdown("#### Input reference")
    st.caption("Every input, what it means and where its default value came from.")
    groups: dict[str, list] = {}
    for name, meta, default in core.input_fields():
        groups.setdefault(meta["group"], []).append((name, meta))
    extra = {"Weights & CG": [("weight_items", {"label": "Component weight table", "unit": "lbf, in"})],
             "Propulsion": [("eta_curve", {"label": "Propeller efficiency table", "unit": "-"})]}
    for gname in core.GROUP_ORDER:
        rows = []
        for name, meta in groups.get(gname, []) + extra.get(gname, []):
            status, xf, desc = core.PARAM_INFO.get(name, ("setting", False, ""))
            u = meta.get("unit") or ""
            rows.append((meta["label"], "" if u in ("", "-") else u, desc,
                         core.STATUS_LABEL[status] + (", XFLR5" if xf else "")))
        with st.expander(tex(gname)):
            st.markdown(GROUP_HELP.get(gname, ""))
            st.markdown(md(pd.DataFrame(rows, columns=["Input", "Unit", "Meaning", "Default from"])))


def md(df: pd.DataFrame, wrap: bool = False) -> str:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "|".join([":--"] * len(df.columns)) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(tex(str(v)) if k == 0 else str(v) for k, v in enumerate(row)) + " |")
    return "\n".join(out)


# ================================================================== Equations
def _eq(title, eqs, where, source):
    st.markdown(f"##### {title}")
    for e in eqs:
        # one relation per line keeps long equations readable on narrow screens
        for part in e.split(r",\qquad "):
            st.latex(part)
    st.markdown(f"<small>where {where}</small>", unsafe_allow_html=True)
    st.markdown(f"<small>Source: {source}</small>", unsafe_allow_html=True)


A1 = "Anderson, *Introduction to Flight*, ch. 6 (Elements of airplane performance)"
A7 = "Anderson, *Introduction to Flight*, ch. 7 (Principles of stability and control)"
APD = "Anderson, *Aircraft Performance and Design*, ch. 6 (Accelerated flight: turns, takeoff and landing)"
R12 = "Raymer, *Aircraft Design: A Conceptual Approach*, ch. 12 (Aerodynamics)"
R16 = "Raymer, *Aircraft Design: A Conceptual Approach*, ch. 16 (Stability, control and handling qualities)"
R6 = "Raymer, *Aircraft Design: A Conceptual Approach*, ch. 6 (Initial sizing), Table 6.4"
NEL = "Nelson, *Flight Stability and Automatic Control*, ch. 2 (Static stability and control)"
FAR = "14 CFR Part 23 before Amendment 23-64 (§23.333 to §23.345), now carried in ASTM F3116"


def equations():
    st.title("Equations")
    st.markdown("Every equation the calculator uses, with its variables and source. The calculator works in imperial "
                "units, with $g$ = 32.174 ft/s² and $\\rho_0$ = 0.0023769 slug/ft³.")

    st.markdown("#### Atmosphere and geometry")
    _eq("Standard atmosphere (troposphere)",
        [r"\rho = \rho_0\left(1 - 6.87559\times10^{-6}\,h\right)^{4.25588},\qquad \sigma = \rho/\rho_0"],
        r"$\rho$ air density, $\rho_0$ sea-level density, $h$ altitude (ft), $\sigma$ density ratio.",
        "U.S. Standard Atmosphere, 1976 (troposphere lapse rate); tabulated in Anderson, *Introduction to Flight*, "
        "Appendix E")
    _eq("Wing geometry and induced drag factor",
        [r"AR = \frac{b^2}{S},\qquad \bar c = \frac{S}{b},\qquad K = \frac{1}{\pi e\,AR}"],
        r"$AR$ aspect ratio, $b$ span, $S$ wing area, $\bar c$ mean chord, $e$ Oswald efficiency, $K$ induced drag factor.",
        A1)
    _eq("Three-dimensional lift-curve slope (Helmbold)",
        [r"a = \frac{\pi AR}{1 + \sqrt{1 + \left(\pi AR / a_0\right)^2}}"],
        r"$a$ wing (or tail) lift slope per radian, $a_0$ airfoil lift slope per radian.",
        R12 + " (lift-curve slope, unswept form)")

    st.markdown("#### Power, climb and ceiling")
    _eq("Level flight",
        [r"C_L = \frac{2W}{\rho V^2 S},\qquad C_D = C_{D,0} + K C_L^2,\qquad D = \tfrac12\rho V^2 S\,C_D,\qquad P_R = D\,V"],
        r"$W$ weight, $V$ true airspeed, $C_L$ lift coefficient, $C_D$ drag coefficient, $C_{D,0}$ zero-lift drag "
        r"coefficient, $D$ drag, $P_R$ power required.", A1)
    _eq("Power available and thrust",
        [r"J = \frac{V}{nD_p},\qquad P_A = \eta_P(J)\,P_{shaft},\qquad T = \frac{\eta_P\,P_{shaft}}{V}"],
        r"$J$ advance ratio, $n$ propeller speed (rev/s), $D_p$ propeller diameter, $\eta_P$ propeller efficiency, "
        r"$P_A$ power available, $T$ thrust.", A1)
    _eq("Engine power at altitude",
        [r"P_{shaft} = P_0\,\sigma"],
        r"$P_0$ sea-level shaft power, $\sigma$ density ratio.", A1)
    _eq("Rate of climb and climb gradient",
        [r"R/C = \frac{P_A - P_R}{W},\qquad \gamma = \frac{T - D}{W}"],
        r"$R/C$ rate of climb, $\gamma$ climb gradient. Part 23 checks use the best $\gamma$ at speeds above $1.1\,V_s$.",
        A1 + "; required gradients from 14 CFR 23.2120")
    _eq("Maximum lift-to-drag ratio",
        [r"\left(\frac{L}{D}\right)_{\max} = \frac{1}{\sqrt{4\,C_{D,0}\,K}},\qquad C_{L} = \sqrt{\frac{C_{D,0}}{K}}"],
        r"$L/D$ lift-to-drag ratio.", A1)
    _eq("Ceilings and time to climb",
        [r"t = \int_0^h \frac{dh}{(R/C)_{\max}}"],
        r"$t$ time to climb. Absolute ceiling where $(R/C)_{\max}$ = 0, service ceiling where it is 100 ft/min.", A1)
    _eq("Stall speed",
        [r"V_s = \sqrt{\frac{2W}{\rho S C_{L,\max}}}"],
        r"$V_s$ stall speed, $C_{L,\max}$ maximum lift coefficient for the configuration.", A1)

    st.markdown("#### Takeoff and landing")
    _eq("Ground effect and best ground-roll lift coefficient",
        [r"\phi = \frac{(16h_w/b)^2}{1 + (16h_w/b)^2},\qquad C_{L,\mathrm{opt}} = \frac{\pi e\,AR\,\mu_r}{2\phi}"],
        r"$\phi$ ground-effect factor on induced drag, $h_w$ wing height above the ground, $\mu_r$ rolling friction coefficient.",
        A1 + " (ground effect); " + APD + " (optimum lift coefficient)")
    _eq("Takeoff ground roll",
        [r"V_{LO} = 1.2\,V_s,\qquad s_{LO} = \frac{1.44\,W^2}{g\,\rho\,S\,C_{L,\max}\left[T - D - \mu_r(W - L)\right]_{0.7V_{LO}}}"],
        r"$V_{LO}$ liftoff speed, $s_{LO}$ ground roll (no obstacle), $T$, $D$ and $L$ thrust, drag and lift at "
        r"$0.7\,V_{LO}$.", A1)
    _eq("Landing ground roll",
        [r"V_{TD} = 1.3\,V_s,\qquad s_L = \frac{1.69\,W^2}{g\,\rho\,S\,C_{L,\max}\left[D + \mu_b(W - L)\right]_{0.7V_{TD}}}"],
        r"$V_{TD}$ touchdown speed, $s_L$ ground roll with zero thrust (no obstacle), $\mu_b$ braking friction coefficient.", A1)
    st.markdown("#### Cruise, range and endurance")
    _eq("Breguet range and endurance (propeller aircraft)",
        [r"R = \frac{\eta_P}{c}\,\frac{C_L}{C_D}\,\ln\frac{W_0}{W_1},\qquad "
         r"E = \frac{\eta_P}{c}\,\frac{C_L^{3/2}}{C_D}\,\sqrt{2\rho S}\left(W_1^{-1/2} - W_0^{-1/2}\right)",
         r"c = \frac{SFC}{550 \times 3600},\qquad \left(\frac{C_L^{3/2}}{C_D}\right)_{\max} = \frac14\left(\frac{3}{K\,C_{D,0}^{1/3}}\right)^{3/4}"],
        r"$R$ range, $E$ endurance, $W_0$ and $W_1$ weight with full and empty tanks, $c$ specific fuel "
        r"consumption per foot (from $SFC$ in lb/(hp·h)).", A1)
    _eq("Best range and endurance speeds",
        [r"V_{(L/D)\max} = \sqrt{\frac{2W}{\rho S}}\left(\frac{K}{C_{D,0}}\right)^{1/4},\qquad "
         r"V_{P_R\,\min} = \sqrt{\frac{2W}{\rho S}}\left(\frac{K}{3C_{D,0}}\right)^{1/4}"],
        r"Minimum drag gives the best range and minimum power the best endurance for a propeller aircraft. "
        r"Range against power setting uses the highest level-flight speed at each setting.", A1)

    st.markdown("#### Turning and glide")
    _eq("Level turn",
        [r"n_{aero} = \frac{q\,C_{L,\max}}{W/S},\qquad n_{power} = \sqrt{\frac{(T - qSC_{D,0})\,qS}{K\,W^2}},\qquad n = \min(n_{aero},\,n_{power},\,n_1)",
         r"R = \frac{V^2}{g\sqrt{n^2 - 1}},\qquad \omega = \frac{g\sqrt{n^2 - 1}}{V}"],
        r"$q$ dynamic pressure, $n$ load factor, $n_1$ structural limit, $R$ turn radius, $\omega$ turn rate.",
        A1 + "; " + APD + " (power-limited turn)")
    _eq("Glide",
        [r"\tan\theta = \frac{1}{L/D},\qquad \text{sink rate} = \frac{D\,V}{W}"],
        r"$\theta$ glide angle.", A1)

    st.markdown("#### Weight, balance and stability")
    _eq("Centre of gravity",
        [r"x_{cg} = \frac{\sum W_i\,x_i}{\sum W_i},\qquad h = \frac{x_{cg} - x_{LEMAC}}{\bar c}"],
        r"$W_i$ and $x_i$ weight and arm of each item, $x_{LEMAC}$ leading edge of the mean aerodynamic chord, "
        r"$h$ CG as a fraction of the mean chord.", "FAA, *Aircraft Weight and Balance Handbook* (FAA-H-8083-1)")
    _eq("Tail volume coefficients",
        [r"V_H = \frac{S_h\,l_h}{S\,\bar c},\qquad V_V = \frac{S_v\,l_v}{S\,b}"],
        r"$S_h$, $S_v$ horizontal and vertical tail areas, $l_h$, $l_v$ tail arms.", R6)
    _eq("Downwash and fuselage contribution",
        [r"\frac{d\varepsilon}{d\alpha} = \frac{2a_w}{\pi AR},\qquad C_{m_\alpha,\mathrm{fus}} = \frac{K_f\,w_f^2\,L_f}{\bar c\,S}\ \ \text{(per degree)}"],
        r"$\varepsilon$ downwash angle at the tail, $a_w$ wing lift slope, $K_f$ empirical factor, $w_f$ fuselage width, "
        r"$L_f$ fuselage length.", NEL + " (downwash); " + R16 + " (fuselage term)")
    _eq("Neutral point and static margin",
        [r"h_n = h_{ac} - \frac{C_{m_\alpha,\mathrm{fus}}}{a_w} + \eta_h V_H \frac{a_t}{a_w}\left(1 - \frac{d\varepsilon}{d\alpha}\right),\qquad SM = h_n - h"],
        r"$h_n$ neutral point, $h_{ac}$ wing aerodynamic centre, $\eta_h$ tail dynamic pressure ratio, $a_t$ tail "
        r"lift slope, $SM$ static margin (all as fractions of $\bar c$).", A7 + "; fuselage term from " + R16)
    _eq("Pitching moment about the CG",
        [r"C_{m,cg} = C_{m,ac} + a_w\alpha_a(h - h_{ac}) + C_{m_\alpha,\mathrm{fus}}\,\alpha_a - \eta_h V_H a_t\,\alpha_t + C_{m_{\delta_e}}\delta_e",
         r"\alpha_a = \alpha - \alpha_{0L},\qquad \alpha_t = \alpha - i_t - \varepsilon,\qquad \varepsilon = \frac{d\varepsilon}{d\alpha}\,\alpha_a"],
        r"$\alpha$ wing angle of attack, $\alpha_a$ angle from zero lift, $\alpha_{0L}$ zero-lift angle, $\alpha_t$ tail "
        r"angle of attack, $i_t$ tail incidence, $\delta_e$ elevator deflection.", A7)
    _eq("Elevator power, trim and hinge moment",
        [r"C_{m_{\delta_e}} = -\eta_h V_H a_t \tau,\qquad \delta_{e,trim} = -\frac{C_{m,cg}(\alpha,\,\delta_e = 0)}{C_{m_{\delta_e}}}",
         r"C_h = C_{h_0} + C_{h_\alpha}\alpha_t + C_{h_\delta}\delta_e,\qquad H = C_h\,q\,S_e\,c_e"],
        r"$\tau$ elevator effectiveness, $C_h$ hinge moment coefficient, $H$ hinge moment, $S_e$ elevator area, "
        r"$c_e$ elevator chord.", NEL)

    st.markdown("#### V-n diagram")
    _eq("Manoeuvre envelope",
        [r"n_1 = \min\left(2.1 + \frac{24000}{W + 10000},\ 3.8\right),\qquad n_2 = -0.4\,n_1",
         r"V_{S1} = \sqrt{\frac{2W}{\rho_0 S C_{L,\max}}},\quad V_A = V_{S1}\sqrt{n_1},\quad V_C = 33\sqrt{W/S}\ \mathrm{(kt)},\quad V_D = 1.4\,V_{C,\min},\quad V_F = 1.8\,V_{S0}"],
        r"$n_1$, $n_2$ positive and negative limit load factors ($W$ in lb, $W/S$ in lb/ft²), $V_A$ manoeuvring "
        r"speed, $V_C$ design cruise speed, $V_D$ design dive speed, $V_F$ design flap speed, $V_{S0}$ full-flap stall speed.",
        FAR + "; " + A1)
    _eq("Gust load factor",
        [r"\mu_g = \frac{2\,W/S}{\rho_0\,\bar c\,a\,g},\qquad K_g = \frac{0.88\,\mu_g}{5.3 + \mu_g},\qquad n = 1 \pm \frac{K_g\,\rho_0\,U_{de}\,V\,a}{2\,W/S}"],
        r"$\mu_g$ mass ratio, $K_g$ gust alleviation factor, $U_{de}$ design gust velocity, $a$ lift slope. "
        r"Ultimate load factor = 1.5 × the largest limit load factor.", FAR + " (§23.341, Pratt formula)")

    st.markdown("#### Wing spar")
    _eq("Schrenk lift distribution",
        [r"L_{net} = n_{ult}\,(W - W_{wing}),\qquad w(y) = \frac12\left[\frac{L_{net}/2}{b/2} + \frac{4\,L_{net}/2}{\pi\,b/2}\sqrt{1 - \left(\frac{y}{b/2}\right)^2}\right]"],
        r"$L_{net}$ net wing load, $n_{ult}$ ultimate load factor, $W_{wing}$ wing weight, $w(y)$ load per unit span "
        r"at distance $y$ from the root (the average of a uniform and an elliptical distribution).",
        "Schrenk, *A Simple Approximation Method for Obtaining the Spanwise Lift Distribution*, NACA TM 948 (1940)")
    _eq("Shear and bending moment",
        [r"V(y) = \int_y^{b/2} w\,d\eta,\qquad M(y) = \int_y^{b/2} V\,d\eta"],
        r"$V$ shear force, $M$ bending moment, $\eta$ integration variable along the span.",
        "Beam statics; Megson, *Aircraft Structures for Engineering Students*")
    _eq("Strut-braced wing",
        [r"R_z = \frac{M(0)}{y_s},\qquad V_s = V - R_z,\quad M_s = M - R_z\,(y_s - y)\quad (y < y_s)",
         r"F_{strut} = R_z\,\frac{L_s}{h_s},\qquad H = R_z\,\frac{y_s}{h_s},\qquad P_{cr} = \frac{\pi^2 E I}{L_s^2}"],
        r"$R_z$ vertical strut reaction (pinned root), $y_s$ strut attachment, $L_s$ strut length, $h_s$ vertical "
        r"drop of the strut, $F_{strut}$ strut force, $H$ axial load in the spar, $P_{cr}$ Euler buckling load.",
        "Statics; Euler buckling from Megson, *Aircraft Structures for Engineering Students*")
    _eq("Spar sizing",
        [r"h_{spar} = (t/c)\,\bar c - 2\,\text{clearance},\qquad A_{cap} = \frac{M}{\sigma_{allow}\,h_{spar}},\qquad t_{web} = \frac{V}{\tau_{allow}\,h_{spar}}"],
        r"$h_{spar}$ spar depth, $A_{cap}$ area of each spar cap, $t_{web}$ web thickness, $\sigma_{allow}$, $\tau_{allow}$ "
        r"allowable stresses.", "Idealised beam bending and shear; Megson, *Aircraft Structures for Engineering Students*")

    st.markdown("#### References")
    st.markdown(
        "- J. D. Anderson, *Introduction to Flight*, McGraw-Hill.\n"
        "- J. D. Anderson, *Aircraft Performance and Design*, McGraw-Hill.\n"
        "- D. P. Raymer, *Aircraft Design: A Conceptual Approach*, AIAA.\n"
        "- R. C. Nelson, *Flight Stability and Automatic Control*, McGraw-Hill.\n"
        "- T. H. G. Megson, *Aircraft Structures for Engineering Students*, Elsevier.\n"
        "- O. Schrenk, NACA Technical Memorandum 948, 1940.\n"
        "- 14 CFR Part 23 (current, and §23.333 to §23.345 before Amendment 23-64); ASTM F3116.\n"
        "- FAA, *Aircraft Weight and Balance Handbook*, FAA-H-8083-1.\n"
        "- Cessna 172S Pilot's Operating Handbook and FAA Type Certificate Data Sheet 3A12 (default inputs).")
