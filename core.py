"""
Aircraft performance calculator (core).

Everything is in imperial units:
    length ft (CG arms in inches, as in a POH), area ft^2, force lbf,
    density slug/ft^3, speed ft/s, power hp, time s, angles deg unless noted.

Usage:
    from core import Inputs, run
    res = run(Inputs())            # Cessna-172-ish baseline
    res = run(Inputs(b=33, S=165)) # tweak anything
    print(res.summary)

Sources: Anderson, "Introduction to Flight" / "Aircraft Performance and Design"
(power curves, takeoff/landing, Breguet, turning, static stability), Raymer,
"Aircraft Design: A Conceptual Approach" (flaps, tail volume coefficients,
fuselage Cm_alpha, landing approach/flare), 14 CFR 23 (Amdt 23-64/65) and the
classic 23.333-23.345 / ASTM F3116 V-n rules.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

# numpy >= 2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# ---------------------------------------------------------------- constants
G = 32.174            # ft/s^2
RHO0 = 0.0023769      # slug/ft^3, sea-level standard
HP = 550.0            # ft*lbf/s per hp
MPH = 1.466667        # ft/s per mph
KT = 1.687810         # ft/s per knot
A0 = 1116.45          # ft/s, sea-level speed of sound
DEG = math.pi / 180.0


def _inp(default, label, unit="", group="", help="", step=None, lo=None, hi=None, options=None):
    """Input field with UI metadata (the interface builds its widgets from this)."""
    return field(default=default, metadata=dict(
        label=label, unit=unit, group=group, help=help, step=step, lo=lo, hi=hi, options=options))


# Generic fixed-pitch propeller efficiency curve. PLACEHOLDER: replace with real prop data.
DEFAULT_ETA_J = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_ETA = [0.0, 0.17, 0.32, 0.46, 0.58, 0.68, 0.75, 0.80, 0.82, 0.80, 0.70, 0.45, 0.0]

# Component weights & arms (inches aft of datum = firewall). PLACEHOLDERS loosely based on a
# C172S: replace with your own weight build-up. Type: empty / crew (always aboard) /
# payload (optional, CG range explores on/off) / fuel (explored from 0 to full).
DEFAULT_WEIGHTS = [
    {"Item": "Engine + prop + mount", "Weight (lbf)": 380.0, "Arm (in)": -20.0, "Type": "empty"},
    {"Item": "Fuselage + interior", "Weight (lbf)": 420.0, "Arm (in)": 60.0, "Type": "empty"},
    {"Item": "Wing (incl. flaps, struts)", "Weight (lbf)": 245.0, "Arm (in)": 42.0, "Type": "empty"},
    {"Item": "Horizontal tail", "Weight (lbf)": 35.0, "Arm (in)": 210.0, "Type": "empty"},
    {"Item": "Vertical tail", "Weight (lbf)": 25.0, "Arm (in)": 200.0, "Type": "empty"},
    {"Item": "Landing gear", "Weight (lbf)": 120.0, "Arm (in)": 30.0, "Type": "empty"},
    {"Item": "Systems, avionics, fuel system", "Weight (lbf)": 438.0, "Arm (in)": 40.0, "Type": "empty"},
    {"Item": "Pilot", "Weight (lbf)": 170.0, "Arm (in)": 37.0, "Type": "crew"},
    {"Item": "Front passenger", "Weight (lbf)": 170.0, "Arm (in)": 37.0, "Type": "payload"},
    {"Item": "Rear passengers (2)", "Weight (lbf)": 340.0, "Arm (in)": 73.0, "Type": "payload"},
    {"Item": "Baggage (area 1)", "Weight (lbf)": 120.0, "Arm (in)": 95.0, "Type": "payload"},
    {"Item": "Fuel (53 gal usable)", "Weight (lbf)": 318.0, "Arm (in)": 48.0, "Type": "fuel"},
]



@dataclass
class Inputs:
    # --- Weights & CG
    W: float = _inp(2550.0, "Max takeoff weight (MTOW)", "lbf", "Weights & CG", "C172S POH: 2,550 lb", lo=500.0, step=10.0)
    W_wing_frac: float = _inp(0.10, "Wing weight / gross weight", "-", "Weights & CG",
                              "Wing weight relieves the spar (net lift = n(W - W_wing))", lo=0.0, hi=0.5, step=0.01)
    x_LEMAC: float = _inp(25.9, "Wing LE of MAC, aft of datum", "in", "Weights & CG",
                          "C172 TCDS 3A12: LEMAC 25.9 in aft of firewall, MAC 58.8 in", step=0.5)
    weight_items: list = field(default_factory=lambda: [dict(d) for d in DEFAULT_WEIGHTS])

    # --- Wing geometry
    b: float = _inp(36.08, "Wing span", "ft", "Wing geometry", "C172S POH: 36 ft 1 in", lo=5.0, step=0.5)
    S: float = _inp(174.0, "Wing area", "ft^2", "Wing geometry", "C172S POH: 174 sq ft", lo=10.0, step=1.0)
    h_wing: float = _inp(7.0, "Wing height above ground", "ft", "Wing geometry",
                         "Estimate for a high wing (C172 overall height 8 ft 11 in)", lo=0.5, step=0.25)

    # --- Aerodynamics
    CD0: float = _inp(0.031, "Zero-lift drag coeff. C_D,0", "-", "Aerodynamics", "UIUC / FlightGear C172 model: 0.031", step=0.0005)
    e: float = _inp(0.79, "Oswald efficiency e", "-", "Aerodynamics", "From UIUC / FlightGear C172 K = 0.054 with AR = 7.48", lo=0.3, hi=1.0, step=0.01)
    CLmax_clean: float = _inp(1.54, "C_L,max clean (no flaps)", "-", "Aerodynamics", "From C172S POH stall speed 53 KCAS at 2,550 lb", step=0.01)
    CL_min: float = _inp(-1.0, "C_L,min (negative stall)", "-", "Aerodynamics",
                         "Negative-g stall line on the V-n diagram", step=0.05)
    a0_wing: float = _inp(5.85, "Wing airfoil lift slope a0 (2-D)", "1/rad", "Aerodynamics",
                          "3-D wing slope is computed from this with Helmbold's equation", step=0.05)
    alpha0L: float = _inp(-2.0, "Wing zero-lift angle alpha_0L", "deg", "Aerodynamics", "NACA 2412 (C172 wing section): about -2 deg", step=0.1)
    Cm_ac: float = _inp(-0.05, "Wing-body Cm_ac", "-", "Aerodynamics", "NACA 2412 ~ -0.05", step=0.005)
    h_ac: float = _inp(0.25, "Wing AC location h_ac", "x/c", "Aerodynamics", step=0.01)

    # --- High-lift (flaps)
    CLmax_TO: float = _inp(1.74, "C_L,max takeoff flaps", "-", "High-lift (flaps)",
                           "Estimate for C172 10 deg flaps", step=0.01)
    CLmax_land: float = _inp(1.88, "C_L,max landing flaps", "-", "High-lift (flaps)",
                             "From C172S POH stall speed 48 KCAS, full flaps", step=0.01)
    dCD0_flaps_TO: float = _inp(0.010, "Delta C_D,0 takeoff flaps", "-", "High-lift (flaps)",
                                "Get from XFLR5 or flap data", step=0.001)
    dCD0_flaps: float = _inp(0.015, "Delta C_D,0 landing flaps", "-", "High-lift (flaps)",
                             "Get from XFLR5 or flap data", step=0.001)

    # --- Tail & stability
    S_h: float = _inp(36.1, "Horizontal tail area S_h", "ft^2", "Tail & stability", "C172: stabilizer 21.56 + elevator 14.53 sq ft (secondary source)", step=0.5)
    b_h: float = _inp(11.3, "Horizontal tail span", "ft", "Tail & stability", "Estimate", step=0.1)
    l_h: float = _inp(14.0, "Tail arm l_h (wing AC to tail AC)", "ft", "Tail & stability", "Estimate from C172 3-view", step=0.1)
    a0_tail: float = _inp(6.0, "Tail airfoil lift slope a0", "1/rad", "Tail & stability", "NACA 0012 ~ 6.0", step=0.05)
    eta_h: float = _inp(0.9, "Tail dynamic pressure ratio eta_h", "-", "Tail & stability", step=0.05)
    i_t: float = _inp(2.0, "Tail incidence i_t (+ = tail LE down)", "deg", "Tail & stability",
                      "Anderson's convention: alpha_t = alpha_wb - i_t - epsilon", step=0.25)
    deps_override: float = _inp(0.0, "Downwash gradient d(eps)/d(alpha) (0 = auto)", "-", "Tail & stability",
                                "Auto = 2 a_w / (pi AR)", step=0.01)
    Kf: float = _inp(0.010, "Fuselage K_f (Raymer Fig. 16.14)", "-", "Tail & stability",
                     "Cm_alpha,fus = K_f w_f^2 L_f / (c S) per deg", step=0.001)
    w_fus: float = _inp(3.7, "Fuselage max width", "ft", "Tail & stability", "Estimate (C172S cabin width 40 in)", step=0.1)
    L_fus: float = _inp(27.17, "Fuselage length", "ft", "Tail & stability", "C172S overall length 27 ft 2 in", step=0.1)
    S_v: float = _inp(18.67, "Vertical tail area S_v", "ft^2", "Tail & stability", "C172: fin 11.24 + rudder 7.43 sq ft (secondary source)", step=0.5)
    l_v: float = _inp(14.5, "Vertical tail arm l_v", "ft", "Tail & stability", "Estimate from C172 3-view", step=0.1)
    VH_target: float = _inp(0.70, "Target horizontal tail volume V_H", "-", "Tail & stability",
                            "Raymer Table 6.4, GA single engine: 0.70", step=0.05)
    VV_target: float = _inp(0.04, "Target vertical tail volume V_V", "-", "Tail & stability",
                            "Raymer Table 6.4, GA single engine: 0.04", step=0.005)
    tau_e: float = _inp(0.5, "Elevator effectiveness tau", "-", "Tail & stability",
                        "From elevator-to-tail chord ratio (~0.5 for c_e/c_t ~ 0.35)", step=0.05)
    S_e: float = _inp(14.53, "Elevator area", "ft^2", "Tail & stability", "C172 (secondary source)", step=0.5)
    c_e: float = _inp(1.3, "Elevator mean chord (aft of hinge)", "ft", "Tail & stability", step=0.05)
    Ch0: float = _inp(0.0, "Hinge moment Ch_0", "-", "Tail & stability", step=0.005)
    Ch_alpha: float = _inp(-0.15, "Hinge moment Ch_alpha", "1/rad", "Tail & stability",
                           "Get from XFLR5 (XFoil hinge moment) or DATCOM", step=0.01)
    Ch_delta: float = _inp(-0.55, "Hinge moment Ch_delta", "1/rad", "Tail & stability",
                           "Get from XFLR5 (XFoil hinge moment) or DATCOM", step=0.01)
    SM_min: float = _inp(0.05, "Minimum static margin", "x/c", "Requirements", step=0.01)
    de_max: float = _inp(25.0, "Max elevator deflection", "deg", "Tail & stability", step=1.0)

    # --- Powerplant
    P_max: float = _inp(180.0, "Max shaft power at S/L", "hp", "Propulsion", "C172S POH: Lycoming IO-360-L2A, 180 BHP", lo=1.0, step=5.0)
    rpm: float = _inp(2700.0, "Engine/prop speed", "rpm", "Propulsion", "C172S POH: rated at 2,700 rpm", step=50.0)
    D_prop: float = _inp(76.0, "Propeller diameter", "in", "Propulsion", "C172S POH: McCauley fixed pitch, 76 in", step=1.0)
    eta_mode: str = _inp("table", "Prop efficiency model", "", "Propulsion",
                         "'table' = eta(J) curve below, 'constant' = single value", options=["table", "constant"])
    eta_const: float = _inp(0.80, "Constant prop efficiency", "-", "Propulsion", lo=0.1, hi=0.95, step=0.01)
    eta_J: list = field(default_factory=lambda: list(DEFAULT_ETA_J))
    eta_vals: list = field(default_factory=lambda: list(DEFAULT_ETA))
    SFC: float = _inp(0.45, "Specific fuel consumption", "lb/(hp*h)", "Propulsion", step=0.01)

    # --- Flight condition / sweep
    altitude: float = _inp(0.0, "Altitude", "ft", "Flight condition", lo=0.0, hi=36000.0, step=500.0)
    V_cruise: float = _inp(209.3, "Cruise speed", "ft/s", "Flight condition", "C172S POH: 124 kt at 75 % power, 8,500 ft", step=1.0)
    V_lo: float = _inp(70.0, "Sweep: min speed", "ft/s", "Flight condition", step=1.0)
    V_hi: float = _inp(350.0, "Sweep: max speed", "ft/s", "Flight condition", step=1.0)
    V_step: float = _inp(1.0, "Sweep: step", "ft/s", "Flight condition", lo=0.1, step=0.5)

    # --- Takeoff & landing
    mu_roll: float = _inp(0.02, "Rolling friction mu_r", "-", "Takeoff & landing", "Hard runway ~0.02", step=0.005)
    mu_brake: float = _inp(0.4, "Braking friction mu_b", "-", "Takeoff & landing", step=0.05)
    VLO_factor: float = _inp(1.2, "V_LO / V_stall", "-", "Takeoff & landing", step=0.05)
    VTD_factor: float = _inp(1.3, "V_TD / V_stall", "-", "Takeoff & landing", step=0.05)
    CL_roll_land: float = _inp(0.0, "Landing ground-roll C_L", "-", "Takeoff & landing",
                               "0 = lift dumped after touchdown (Anderson's assumption)", step=0.05)
    runway_factor: float = _inp(25.0, "Maximum ground roll / wingspan", "-", "Requirements",
                                "Set to 0 to skip this check", lo=0.0, step=1.0)
    mach_max: float = _inp(0.30, "Maximum Mach number", "-", "Requirements",
                           "Above about 0.3 the incompressible-flow equations lose accuracy", step=0.01)
    grad_TO_req: float = _inp(0.083, "Required takeoff climb gradient", "-", "Requirements",
                              "14 CFR 23.2120(a)(1): 8.3% levels 1-2 low-speed landplanes", step=0.001)
    grad_balked_req: float = _inp(0.03, "Required balked-landing gradient", "-", "Requirements",
                                  "14 CFR 23.2120(c): 3%, landing flaps", step=0.001)

    # --- Loads (V-n)
    n1_cap: float = _inp(3.8, "Max positive limit n (cap)", "-", "Loads (V-n)",
                         "n1 = 2.1 + 24000/(W+10000), need not exceed this", step=0.1)
    n_neg_ratio: float = _inp(0.4, "|n2| / n1", "-", "Loads (V-n)", step=0.05)
    Ude_C: float = _inp(50.0, "Gust velocity at V_C", "ft/s", "Loads (V-n)", step=1.0)
    Ude_D: float = _inp(25.0, "Gust velocity at V_D", "ft/s", "Loads (V-n)", step=1.0)
    VC_coef: float = _inp(33.0, "k in V_C,min = k sqrt(W/S) [kt]", "-", "Loads (V-n)", step=0.5)
    VD_basis: str = _inp("Part 23: 1.4 x V_C,min", "V_D basis", "", "Loads (V-n)",
                         "Part 23 rule, or a chosen factor times the cruise speed",
                         options=["Part 23: 1.4 x V_C,min", "Factor x cruise speed"])
    VD_ratio: float = _inp(1.4, "V_D / V_C,min (Part 23 basis)", "-", "Loads (V-n)", step=0.05)
    VD_cruise_factor: float = _inp(1.3, "V_D / V_cruise (cruise-factor basis)", "-", "Loads (V-n)", step=0.05)
    VF_ratio: float = _inp(1.8, "V_F / V_S0", "-", "Loads (V-n)", step=0.05)
    FoS: float = _inp(1.5, "Factor of safety (ultimate/limit)", "-", "Loads (V-n)", step=0.05)

    # --- Spar
    n_stations: int = _inp(41, "Spanwise stations (half span)", "-", "Spar", "More points give smoother results", lo=5, hi=501, step=1)
    strut: bool = _inp(True, "Strut-braced wing", "", "Spar", "Root pinned; strut reaction makes root moment zero")
    y_strut: float = _inp(8.42, "Strut attach point from root", "ft", "Spar", step=0.1)
    h_strut: float = _inp(4.5, "Strut: vertical drop wing to fuselage attach", "ft", "Spar",
                          "Sets strut angle -> strut force and spar axial load", step=0.1)
    x_front: float = _inp(0.25, "Front spar x/c", "-", "Spar", step=0.01)
    x_rear: float = _inp(0.75, "Rear spar x/c", "-", "Spar", step=0.01)
    share_front: float = _inp(0.80, "Front spar load share", "-", "Spar",
                              "The two shares may add to more than 1 for a conservative design", step=0.05)
    share_rear: float = _inp(0.45, "Rear spar load share", "-", "Spar", step=0.05)
    tc_front: float = _inp(0.12, "Airfoil t/c at front spar", "-", "Spar", step=0.005)
    tc_rear: float = _inp(0.08, "Airfoil t/c at rear spar", "-", "Spar", step=0.005)
    clearance: float = _inp(0.5, "Skin/rib clearance (each side)", "in", "Spar", step=0.05)
    sigma_allow: float = _inp(64000.0, "Spar cap allowable stress (ultimate)", "psi", "Spar",
                              "2024-T3 F_tu ~ 64 ksi; check your material/source", step=1000.0)
    tau_allow: float = _inp(40000.0, "Spar web allowable shear (ultimate)", "psi", "Spar",
                            "2024-T3 F_su ~ 40 ksi; check your source", step=1000.0)
    E_strut: float = _inp(10.5e6, "Strut Young's modulus", "psi", "Spar", "Aluminium ~10.5e6", step=1e5)

    # ---- helpers
    @classmethod
    def from_dict(cls, d: dict) -> "Inputs":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, path: str) -> "Inputs":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_json(self, path: str | None = None) -> str:
        s = json.dumps(asdict(self), indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(s)
        return s




# ---------------------------------------------------------------- input descriptions and data status
# status: published (Cessna, NACA or regulation data), derived (calculated from published data),
#         estimate (reasonable guess), placeholder (stand-in value, replace it), setting (method or design choice)
# xflr5:  True if an XFLR5 analysis of your own design should supply this value
P, D_, E_, PH, ST = "published", "derived", "estimate", "placeholder", "setting"
PARAM_INFO = {
    "W": (P, False, "Maximum takeoff weight. Most results are calculated at this weight."),
    "W_wing_frac": (E_, False, "Wing weight as a fraction of MTOW. The wing's own weight reduces the net load on the spar."),
    "x_LEMAC": (P, False, "Distance from the datum to the leading edge of the mean aerodynamic chord. Converts CG positions to % MAC."),
    "weight_items": (PH, False, "Component weights and their distances from the datum. The total empty weight matches the Cessna 172S; the split between components is a placeholder."),
    "b": (P, False, "Wing span, tip to tip."),
    "S": (P, False, "Wing reference area."),
    "h_wing": (E_, False, "Height of the wing above the runway. Sets the ground effect during takeoff and landing."),
    "CD0": (P, True, "Drag coefficient at zero lift (parasite drag) for the whole aircraft."),
    "e": (D_, True, "Oswald efficiency. Sets how quickly drag grows with lift (induced drag)."),
    "CLmax_clean": (D_, True, "Maximum lift coefficient with flaps up. Sets the clean stall speed."),
    "CL_min": (PH, True, "Most negative lift coefficient. Sets the negative stall line on the V-n diagram."),
    "a0_wing": (P, True, "Lift-curve slope of the wing's airfoil section (2-D). The 3-D wing slope is calculated from it."),
    "alpha0L": (P, True, "Angle of attack at which the wing makes no lift."),
    "Cm_ac": (P, True, "Pitching moment coefficient of the wing and fuselage about the aerodynamic centre."),
    "h_ac": (ST, True, "Aerodynamic centre position as a fraction of the mean chord (0.25 for thin wings)."),
    "CLmax_TO": (E_, True, "Maximum lift coefficient with flaps in the takeoff setting."),
    "CLmax_land": (D_, True, "Maximum lift coefficient with flaps fully down. Sets the landing stall speed."),
    "dCD0_flaps_TO": (PH, True, "Extra zero-lift drag with takeoff flaps."),
    "dCD0_flaps": (PH, True, "Extra zero-lift drag with landing flaps."),
    "S_h": (P, False, "Horizontal tail area (stabilizer plus elevator)."),
    "b_h": (E_, False, "Horizontal tail span. Sets the tail aspect ratio and lift slope."),
    "l_h": (E_, False, "Distance from the wing's aerodynamic centre to the tail's aerodynamic centre."),
    "a0_tail": (P, True, "Lift-curve slope of the tail's airfoil section (2-D)."),
    "eta_h": (ST, True, "Dynamic pressure at the tail divided by the free-stream value (losses from the wing wake)."),
    "i_t": (PH, False, "Tail incidence relative to the wing. Positive means the tail leading edge points down."),
    "deps_override": (ST, True, "Rate at which downwash at the tail changes with angle of attack. Leave at 0 to calculate it."),
    "Kf": (E_, False, "Empirical fuselage factor from Raymer's chart. Sets how much the fuselage reduces stability."),
    "w_fus": (E_, False, "Maximum fuselage width."),
    "L_fus": (P, False, "Fuselage length."),
    "S_v": (P, False, "Vertical tail area (fin plus rudder)."),
    "l_v": (E_, False, "Distance from the wing to the vertical tail's aerodynamic centre."),
    "VH_target": (ST, False, "Typical horizontal tail volume for comparison (Raymer: 0.70 for single-engine light aircraft)."),
    "VV_target": (ST, False, "Typical vertical tail volume for comparison (Raymer: 0.04 for single-engine light aircraft)."),
    "tau_e": (E_, True, "Elevator effectiveness: how much a degree of elevator changes the tail's angle of attack."),
    "S_e": (P, False, "Elevator area (aft of the hinge)."),
    "c_e": (E_, False, "Mean elevator chord aft of the hinge. Converts the hinge moment coefficient to a moment."),
    "Ch0": (PH, True, "Elevator hinge moment coefficient at zero tail angle and zero deflection."),
    "Ch_alpha": (PH, True, "Change of hinge moment coefficient with tail angle of attack."),
    "Ch_delta": (PH, True, "Change of hinge moment coefficient with elevator deflection."),
    "SM_min": (ST, False, "Smallest acceptable static margin. Used for the stability check and the aft CG limit."),
    "de_max": (ST, False, "Largest elevator deflection available. Used for the trim check."),
    "P_max": (P, False, "Engine shaft power at sea level."),
    "rpm": (P, False, "Engine and propeller speed (direct drive)."),
    "D_prop": (P, False, "Propeller diameter."),
    "eta_mode": (ST, False, "Use a propeller efficiency curve against advance ratio, or one constant efficiency."),
    "eta_const": (E_, False, "Propeller efficiency used when the model is 'constant'."),
    "eta_curve": (PH, False, "Propeller efficiency against advance ratio J. The default is a generic fixed-pitch curve."),
    "SFC": (E_, False, "Fuel burned per horsepower per hour. Used for range and endurance."),
    "altitude": (ST, False, "Altitude for the power, cruise, turn and glide results. Takeoff and landing also use it."),
    "V_cruise": (P, False, "Cruise speed. Used for cruise range, trim at cruise and hinge moments."),
    "V_lo": (ST, False, "Lowest speed in the speed tables."),
    "V_hi": (ST, False, "Highest speed in the speed tables."),
    "V_step": (ST, False, "Speed step in the speed tables."),
    "mu_roll": (ST, False, "Rolling friction coefficient during the takeoff run (about 0.02 on a paved runway)."),
    "mu_brake": (ST, False, "Braking friction coefficient during the landing run."),
    "VLO_factor": (ST, False, "Liftoff speed as a multiple of stall speed."),
    "VTD_factor": (ST, False, "Touchdown speed as a multiple of stall speed."),
    "CL_roll_land": (ST, False, "Lift coefficient during the landing run (0 assumes lift is dumped)."),
    "runway_factor": (ST, False, "Longest acceptable ground roll, as a multiple of wingspan. 0 skips the check."),
    "mach_max": (ST, False, "Highest acceptable Mach number for the incompressible equations."),
    "grad_TO_req": (ST, False, "Minimum climb gradient after takeoff (Part 23: 8.3 % for small low-speed landplanes)."),
    "grad_balked_req": (ST, False, "Minimum climb gradient in the landing configuration (Part 23: 3 %)."),
    "n1_cap": (ST, False, "Upper cap on the positive limit load factor (3.8 for normal category)."),
    "n_neg_ratio": (ST, False, "Negative limit load factor as a fraction of the positive one (0.4 in Part 23)."),
    "Ude_C": (ST, False, "Design gust velocity at cruise speed (50 ft/s in Part 23)."),
    "Ude_D": (ST, False, "Design gust velocity at dive speed (25 ft/s in Part 23)."),
    "VC_coef": (ST, False, "Factor in the minimum design cruise speed (33 in Part 23)."),
    "VD_basis": (ST, False, "How the design dive speed is set."),
    "VD_ratio": (ST, False, "Dive speed divided by minimum cruise speed (1.4 in Part 23)."),
    "VD_cruise_factor": (ST, False, "Dive speed divided by cruise speed, when that basis is chosen."),
    "VF_ratio": (ST, False, "Design flap speed divided by the full-flap stall speed (1.8 in Part 23)."),
    "FoS": (ST, False, "Ultimate load divided by limit load (1.5 in Part 23)."),
    "n_stations": (ST, False, "Number of points along the half span for the spar loads."),
    "strut": (ST, False, "Wing braced by a lift strut (root pinned) instead of a cantilever wing."),
    "y_strut": (E_, False, "Distance from the wing root to the strut attachment."),
    "h_strut": (E_, False, "Vertical distance from the strut's wing attachment to its fuselage attachment."),
    "x_front": (ST, False, "Front spar position as a fraction of chord."),
    "x_rear": (ST, False, "Rear spar position as a fraction of chord."),
    "share_front": (E_, False, "Fraction of the wing bending load the front spar is sized for."),
    "share_rear": (E_, False, "Fraction of the wing bending load the rear spar is sized for."),
    "tc_front": (P, False, "Airfoil thickness at the front spar, as a fraction of chord."),
    "tc_rear": (E_, False, "Airfoil thickness at the rear spar, as a fraction of chord."),
    "clearance": (E_, False, "Space lost to skin and ribs above and below each spar."),
    "sigma_allow": (E_, False, "Allowable stress in the spar caps at ultimate load (typical aluminium alloy)."),
    "tau_allow": (E_, False, "Allowable shear stress in the spar web at ultimate load (typical aluminium alloy)."),
    "E_strut": (E_, False, "Young's modulus of the strut material (aluminium)."),
}
STATUS_LABEL = {P: "Published", D_: "Derived", E_: "Estimate", PH: "Placeholder", ST: "Setting"}
GROUP_ORDER = ["Weights & CG", "Wing geometry", "Aerodynamics", "High-lift (flaps)", "Propulsion", "Tail & stability",
               "Flight condition", "Takeoff & landing", "Loads (V-n)", "Spar", "Requirements"]


def input_fields():
    """(name, metadata, default) for every UI-editable scalar input, in display-group order.
    metadata also carries 'status', 'xflr5' and 'desc' from PARAM_INFO."""
    out = []
    for f in fields(Inputs):
        if f.metadata.get("label"):
            meta = dict(f.metadata)
            st_, xf, desc = PARAM_INFO.get(f.name, (ST, False, ""))
            meta.update(status=st_, xflr5=xf, desc=desc)
            out.append((f.name, meta, f.default))
    order = {g: k for k, g in enumerate(GROUP_ORDER)}
    return sorted(out, key=lambda t: order.get(t[1]["group"], 99))


# ---------------------------------------------------------------- atmosphere
def density(h_ft: float | np.ndarray) -> float | np.ndarray:
    """ISA density (troposphere, h <= 36,089 ft). Matches Anderson App. E table."""
    h = np.asarray(h_ft, dtype=float)
    return RHO0 * (1.0 - 6.87559e-6 * h) ** 4.25588


# ---------------------------------------------------------------- building blocks
def AR(i: Inputs) -> float:
    return i.b ** 2 / i.S


def chord(i: Inputs) -> float:
    return i.S / i.b


def K_ind(i: Inputs) -> float:
    """Induced-drag factor K = 1/(pi e AR)."""
    return 1.0 / (math.pi * i.e * AR(i))


def helmbold(a0: float, ar: float) -> float:
    """3-D lift-curve slope (1/rad) of a straight wing from its airfoil slope (Helmbold)."""
    k = math.pi * ar / a0
    return math.pi * ar / (1.0 + math.sqrt(1.0 + k * k))


def a_wing(i: Inputs) -> float:
    return helmbold(i.a0_wing, AR(i))


def ground_effect_phi(i: Inputs) -> float:
    x = (16.0 * i.h_wing / i.b) ** 2
    return x / (1.0 + x)


def stall_speed(W, rho, S, CLmax):
    return np.sqrt(2.0 * W / (rho * S * CLmax))


def prop_eta(i: Inputs, J):
    J = np.asarray(J, dtype=float)
    if i.eta_mode == "constant":
        return np.full_like(J, i.eta_const)
    xs, ys = np.asarray(i.eta_J, float), np.asarray(i.eta_vals, float)
    order = np.argsort(xs)
    return np.clip(np.interp(J, xs[order], ys[order], left=ys[order][0], right=0.0), 0.0, 1.0)


def advance_ratio(i: Inputs, V):
    return np.asarray(V, float) / (i.rpm / 60.0 * i.D_prop / 12.0)


def shaft_power_hp(i: Inputs, rho):
    """Shaft power falls with density ratio (Anderson)."""
    return i.P_max * rho / RHO0


def power_available_hp(i: Inputs, V, rho):
    return prop_eta(i, advance_ratio(i, V)) * shaft_power_hp(i, rho)


def thrust(i: Inputs, V, rho):
    """Thrust T = eta P / V (lbf). Finite at V -> 0 because eta ~ J there."""
    V = np.maximum(np.asarray(V, float), 1e-6)
    P = shaft_power_hp(i, rho) * HP
    Vs = np.maximum(V, 1.0)
    return prop_eta(i, advance_ratio(i, Vs)) * P / Vs


# ---------------------------------------------------------------- power curves
def power_table(i: Inputs, rho: float | None = None, W: float | None = None,
                V: np.ndarray | None = None, CD0: float | None = None, CLmax: float | None = None) -> pd.DataFrame:
    """Level-flight power required / available vs speed ."""
    rho = density(i.altitude) if rho is None else rho
    W = i.W if W is None else W
    CD0 = i.CD0 if CD0 is None else CD0
    CLmax = i.CLmax_clean if CLmax is None else CLmax
    if V is None:
        V = np.arange(i.V_lo, i.V_hi + 1e-9, i.V_step)
    V = np.asarray(V, float)
    K = K_ind(i)
    q = 0.5 * rho * V ** 2
    CL = W / (q * i.S)
    CDi = K * CL ** 2
    CD = CD0 + CDi
    D = q * i.S * CD
    PR = D * V
    J = advance_ratio(i, V)
    eta = prop_eta(i, J)
    PA = eta * shaft_power_hp(i, rho)
    excess = PA - PR / HP
    RC = excess * HP / W
    return pd.DataFrame({
        "V (ft/s)": V, "V (mph)": V / MPH, "V (kt)": V / KT,
        "C_L": CL, "C_D,i": CDi, "C_D": CD, "L/D": CL / CD, "Drag (lbf)": D,
        "P_R (ft*lbf/s)": PR, "P_R (hp)": PR / HP, "J": J, "eta_P": eta,
        "P_A (hp)": PA, "Thrust (lbf)": PA * HP / V, "Excess P (hp)": excess,
        "R/C (ft/s)": RC, "R/C (ft/min)": RC * 60.0, "Climb gradient (%)": 100 * RC / V,
        "Mach (sea-level a)": V / A0, "Stalled": CL > CLmax,
    })


def _vmax_from_table(df: pd.DataFrame) -> float:
    """Highest speed where P_A >= P_R (linear interpolation on excess power)."""
    ok = df[~df["Stalled"]]
    V, ex = ok["V (ft/s)"].to_numpy(), ok["Excess P (hp)"].to_numpy()
    if len(V) == 0 or ex.max() < 0:
        return float("nan")
    idx = np.where((ex[:-1] >= 0) & (ex[1:] < 0))[0]
    if len(idx) == 0:
        return float(V[-1]) if ex[-1] >= 0 else float("nan")
    k = idx[-1]
    return float(V[k] + ex[k] * (V[k + 1] - V[k]) / (ex[k] - ex[k + 1]))


def max_climb(i: Inputs, rho: float, W: float | None = None) -> tuple[float, float]:
    """(max R/C ft/s, speed ft/s) at density rho, using a fine speed sweep."""
    W = i.W if W is None else W
    Vs = float(stall_speed(W, rho, i.S, i.CLmax_clean))
    V = np.linspace(Vs * 1.0001, max(Vs * 4, i.V_hi), 600)
    df = power_table(i, rho=rho, W=W, V=V)
    k = int(df["R/C (ft/s)"].idxmax())
    return float(df.at[k, "R/C (ft/s)"]), float(df.at[k, "V (ft/s)"])


def climb_table(i: Inputs, h_max: float = 25000.0, dh: float = 500.0) -> tuple[pd.DataFrame, dict]:
    hs = np.arange(0.0, h_max + 1e-9, dh)
    rows = []
    for h in hs:
        rc, v = max_climb(i, float(density(h)))
        rows.append((h, float(density(h)), rc * 60.0, v, v / KT))
    df = pd.DataFrame(rows, columns=["Altitude (ft)", "rho (slug/ft^3)", "Max R/C (ft/min)",
                                     "V for max R/C (ft/s)", "V for max R/C (kt)"])

    def _cross(level):
        r = df["Max R/C (ft/min)"].to_numpy()
        idx = np.where((r[:-1] >= level) & (r[1:] < level))[0]
        if len(idx) == 0:
            return float("nan")
        k = idx[0]
        return float(hs[k] + (r[k] - level) * (hs[k + 1] - hs[k]) / (r[k] - r[k + 1]))

    rc_s = df["Max R/C (ft/min)"].to_numpy() / 60.0
    t = np.zeros_like(hs)
    for k in range(1, len(hs)):
        if rc_s[k] <= 0 or rc_s[k - 1] <= 0:
            t[k:] = np.nan
            break
        t[k] = t[k - 1] + 0.5 * (1 / rc_s[k] + 1 / rc_s[k - 1]) * (hs[k] - hs[k - 1])
    df["Time to climb (min)"] = t / 60.0
    return df, {"absolute_ceiling_ft": _cross(0.0), "service_ceiling_ft": _cross(100.0)}


def climb_gradient(i: Inputs, CD0: float, CLmax: float, rho: float = RHO0) -> dict:
    """Best steady climb gradient (T - D)/W at full power for a configuration (out of ground effect)."""
    Vs = float(stall_speed(i.W, rho, i.S, CLmax))
    V = np.linspace(1.1 * Vs, max(3.5 * Vs, 1.1 * Vs + 10), 500)
    q = 0.5 * rho * V ** 2
    CL = i.W / (q * i.S)
    D = q * i.S * (CD0 + K_ind(i) * CL ** 2)
    grad = (thrust(i, V, rho) - D) / i.W
    k = int(np.argmax(grad))
    return {"Max gradient (%)": 100 * grad[k], "at V (kt)": V[k] / KT, "V_stall config (kt)": Vs / KT}


# ---------------------------------------------------------------- takeoff / landing
def takeoff(i: Inputs, CLmax: float, flaps: bool) -> dict:
    """Takeoff ground roll, no obstacle (Anderson, 0.7 V_LO average-force method)."""
    rho = float(density(i.altitude))
    W, S, K = i.W, i.S, K_ind(i)
    phi = ground_effect_phi(i)
    CD0 = i.CD0 + (i.dCD0_flaps_TO if flaps else 0.0)
    Vs = float(stall_speed(W, rho, S, CLmax))
    VLO = i.VLO_factor * Vs
    CL_g = min(math.pi * i.e * AR(i) * i.mu_roll / (2 * phi), CLmax)   # lift coefficient for the shortest run
    V7 = 0.7 * VLO
    q = 0.5 * rho * V7 ** 2
    T7 = float(thrust(i, V7, rho))
    D7 = q * S * (CD0 + phi * K * CL_g ** 2)
    L7 = q * S * CL_g
    F7 = T7 - D7 - i.mu_roll * (W - L7)
    s_LO = 1.44 * W ** 2 / (G * rho * S * CLmax * F7) if F7 > 0 else float("inf")
    return {"V_stall (ft/s)": Vs, "V_LO (ft/s)": VLO, "V_LO (kt)": VLO / KT, "phi (ground effect)": phi,
            "C_L ground roll": CL_g, "Thrust at 0.7 V_LO (lbf)": T7, "Ground roll (ft)": s_LO}


def landing(i: Inputs, CLmax: float, flaps: bool) -> dict:
    """Landing ground roll, no obstacle, zero thrust (Anderson, 0.7 V_TD average-force method)."""
    rho = float(density(i.altitude))
    W, S, K = i.W, i.S, K_ind(i)
    phi = ground_effect_phi(i)
    CD0 = i.CD0 + (i.dCD0_flaps if flaps else 0.0)
    Vs = float(stall_speed(W, rho, S, CLmax))
    VT = i.VTD_factor * Vs
    V7 = 0.7 * VT
    q = 0.5 * rho * V7 ** 2
    L = q * S * i.CL_roll_land
    D = q * S * (CD0 + phi * K * i.CL_roll_land ** 2)
    sL = 1.69 * W ** 2 / (G * rho * S * CLmax * (D + i.mu_brake * (W - L)))
    return {"V_stall (ft/s)": Vs, "V_TD (ft/s)": VT, "V_TD (kt)": VT / KT, "Ground roll (ft)": sL}


# ---------------------------------------------------------------- cruise / range
def range_endurance(i: Inputs, W_fuel: float) -> tuple[dict, pd.DataFrame]:
    rho = float(density(i.altitude))
    W0, W1 = i.W, i.W - W_fuel
    K, CD0, S = K_ind(i), i.CD0, i.S
    c = i.SFC / (HP * 3600.0)  # 1/ft
    LD_max = 1.0 / math.sqrt(4 * CD0 * K)
    CL32_CD_max = 0.25 * (3.0 / (K * CD0 ** (1 / 3))) ** 0.75
    V_LDmax = math.sqrt(2 * W0 / (rho * S)) * (K / CD0) ** 0.25
    V_minPR = math.sqrt(2 * W0 / (rho * S)) * (K / (3 * CD0)) ** 0.25
    eta_R = float(prop_eta(i, advance_ratio(i, V_LDmax)))
    eta_E = float(prop_eta(i, advance_ratio(i, V_minPR)))
    ln = math.log(W0 / W1) if 0 < W1 < W0 else float("nan")
    R_max = eta_R / c * LD_max * ln
    E_max = eta_E / c * CL32_CD_max * math.sqrt(2 * rho * S) * (W1 ** -0.5 - W0 ** -0.5) if W1 > 0 else float("nan")

    pt = power_table(i, rho=rho, V=np.array([i.V_cruise]))
    LD_c, eta_c = float(pt["L/D"].iloc[0]), float(pt["eta_P"].iloc[0])
    R_c = eta_c / c * LD_c * ln
    pct_c = float(pt["P_R (hp)"].iloc[0] / eta_c / shaft_power_hp(i, rho) * 100) if eta_c > 0 else float("nan")
    summary = {
        "Fuel used for range (lbf)": W_fuel,
        "(L/D)max": LD_max, "C_L at (L/D)max": math.sqrt(CD0 / K),
        "V best range, min drag (ft/s)": V_LDmax, "V best range (kt)": V_LDmax / KT,
        "V best endurance, min P_R (ft/s)": V_minPR, "V best endurance (kt)": V_minPR / KT,
        "Max range (nmi)": R_max / 6076.12, "Max endurance (h)": E_max / 3600.0,
        "Cruise L/D": LD_c, "Cruise power setting (%)": pct_c,
        "Range at cruise speed (nmi)": R_c / 6076.12,
        "Endurance at cruise speed (h)": R_c / i.V_cruise / 3600.0,
    }
    P_shaft = shaft_power_hp(i, rho)
    df = power_table(i, rho=rho, V=np.arange(max(i.V_lo, 30.0), i.V_hi + 1e-9, 0.25))
    rows = []
    for pct in range(40, 101, 5):
        ex = df["eta_P"] * P_shaft * pct / 100 - df["P_R (hp)"]
        ok = (~df["Stalled"]).to_numpy() & (ex.to_numpy() >= 0)
        if not ok.any():
            rows.append((pct, *[np.nan] * 6)); continue
        k = np.where(ok)[0][-1]
        V, LD, eta = df["V (ft/s)"].iloc[k], df["L/D"].iloc[k], df["eta_P"].iloc[k]
        R = eta / c * LD * ln
        rows.append((pct, V, V / KT, LD, i.SFC * P_shaft * pct / 100, R / 6076.12, R / V / 3600.0))
    tbl = pd.DataFrame(rows, columns=["Power setting (%)", "V (ft/s)", "V (kt)", "L/D",
                                      "Fuel flow (lb/h)", "Range (nmi)", "Endurance (h)"])
    return summary, tbl


# ---------------------------------------------------------------- turning / glide
def turn_table(i: Inputs) -> tuple[pd.DataFrame, dict]:
    rho = float(density(i.altitude))
    W, S, K, CD0 = i.W, i.S, K_ind(i), i.CD0
    n1 = min(2.1 + 24000.0 / (i.W + 10000.0), i.n1_cap)
    V = np.arange(i.V_lo, i.V_hi + 1e-9, i.V_step)
    q = 0.5 * rho * V ** 2
    n_aero = q * i.CLmax_clean / (W / S)
    T = power_available_hp(i, V, rho) * HP / V
    n_pow = np.sqrt(np.maximum((T - q * S * CD0) * q * S / (K * W ** 2), 0.0))
    n = np.minimum.reduce([n_aero, n_pow, np.full_like(V, n1)])
    limit = np.select([n == n_aero, n == n_pow], ["C_L,max", "power"], "structure")
    root = np.sqrt(np.maximum(n ** 2 - 1, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(root > 0, V ** 2 / (G * root), np.nan)
        omega = np.where(root > 0, np.degrees(G * root / V), np.nan)
    df = pd.DataFrame({"V (ft/s)": V, "V (kt)": V / KT, "n aero limit": n_aero, "n power limit": n_pow,
                       "n structural": n1, "n max": n, "Limited by": limit,
                       "Bank angle (deg)": np.degrees(np.arccos(np.clip(1 / np.maximum(n, 1), -1, 1))),
                       "Turn radius (ft)": R, "Turn rate (deg/s)": omega})
    s = {}
    if df["Turn radius (ft)"].notna().any():
        kr = df["Turn radius (ft)"].idxmin(); kw = df["Turn rate (deg/s)"].idxmax()
        s = {"Min turn radius (ft)": df.at[kr, "Turn radius (ft)"], "V for min radius (kt)": df.at[kr, "V (kt)"],
             "Max turn rate (deg/s)": df.at[kw, "Turn rate (deg/s)"], "V for max rate (kt)": df.at[kw, "V (kt)"]}
    return df, s


def glide(i: Inputs) -> tuple[pd.DataFrame, dict]:
    rho = float(density(i.altitude))
    out, summ = [], {}
    K = K_ind(i)
    for label, CD0, CLmax in [("clean", i.CD0, i.CLmax_clean), ("flaps", i.CD0 + i.dCD0_flaps, i.CLmax_land)]:
        df = power_table(i, rho=rho, CD0=CD0, CLmax=CLmax)
        df = df[~df["Stalled"]]
        out.append(pd.DataFrame({"Config": label, "V (ft/s)": df["V (ft/s)"], "V (kt)": df["V (kt)"],
                                 "L/D": df["L/D"], "Glide angle (deg)": np.degrees(np.arctan(1 / df["L/D"])),
                                 "Sink rate (ft/min)": df["P_R (ft*lbf/s)"] / i.W * 60}))
        LD = 1 / math.sqrt(4 * CD0 * K)
        Vbg = math.sqrt(2 * i.W / (rho * i.S * math.sqrt(CD0 / K)))
        Vms = math.sqrt(2 * i.W / (rho * i.S * math.sqrt(3 * CD0 / K)))
        Vstall = float(stall_speed(i.W, rho, i.S, CLmax))
        CL_ms = math.sqrt(3 * CD0 / K)
        sink_min = math.sqrt(2 * i.W / (rho * i.S)) / (CL_ms ** 1.5 / (CD0 + K * CL_ms ** 2))
        summ[label] = {"(L/D)max": LD, "Min glide angle (deg)": math.degrees(math.atan(1 / LD)),
                       "Best-glide V (kt)": Vbg / KT, "Glide ratio (ft fwd per ft lost)": LD,
                       "Min-sink V (kt)": Vms / KT, "Min sink (ft/min)": sink_min * 60,
                       "Stall V (kt)": Vstall / KT,
                       "Note": "best-glide/min-sink below stall -> fly just above stall" if Vms < Vstall else ""}
    return pd.concat(out, ignore_index=True), summ


# ---------------------------------------------------------------- weights & CG
def weights_cg(i: Inputs) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Empty weight/CG, weight limits and CG range from all loading combinations up to MTOW."""
    items = pd.DataFrame(i.weight_items)
    if items.empty:
        items = pd.DataFrame(columns=["Item", "Weight (lbf)", "Arm (in)", "Type"])
    items["Weight (lbf)"] = pd.to_numeric(items["Weight (lbf)"], errors="coerce").fillna(0.0)
    items["Arm (in)"] = pd.to_numeric(items["Arm (in)"], errors="coerce").fillna(0.0)
    items["Type"] = items["Type"].fillna("empty").str.lower()
    items["Moment (lbf*in)"] = items["Weight (lbf)"] * items["Arm (in)"]
    c_in = chord(i) * 12
    pct = lambda x: 100 * (x - i.x_LEMAC) / c_in

    def wm(df):
        return float(df["Weight (lbf)"].sum()), float(df["Moment (lbf*in)"].sum())

    We, Me = wm(items[items["Type"] == "empty"])
    Wc, Mc = wm(items[items["Type"] == "crew"])
    fuel = items[items["Type"] == "fuel"]
    Wf, Mf = wm(fuel)
    arm_f = Mf / Wf if Wf > 0 else 0.0
    pay = items[items["Type"] == "payload"].reset_index(drop=True)

    cases = []
    for mask in product([0, 1], repeat=len(pay)):
        sel = pay[np.array(mask, bool)] if len(pay) else pay
        Wp, Mp = wm(sel) if len(sel) else (0.0, 0.0)
        W0, M0 = We + Wc + Wp, Me + Mc + Mp
        if W0 > i.W + 1e-6:
            continue
        for fuel_w in sorted({0.0, min(Wf, i.W - W0)}):
            W = W0 + fuel_w
            cg = (M0 + fuel_w * arm_f) / W if W > 0 else float("nan")
            name = " + ".join(["crew"] + list(sel["Item"])) if len(sel) else "crew only"
            cases.append({"Loading": name, "Fuel (lbf)": fuel_w, "Weight (lbf)": W, "CG (in)": cg, "CG (%MAC)": pct(cg)})
    cases = pd.DataFrame(cases)
    s = {"Empty weight (lbf)": We, "Empty CG (in)": Me / We if We else float("nan"),
         "MTOW (lbf)": i.W, "Useful load (lbf)": i.W - We, "Full fuel (lbf)": Wf,
         "Payload with full fuel (lbf)": i.W - We - Wc - Wf,
         "Total payload items (lbf)": float(pay["Weight (lbf)"].sum()),
         "Fuel with all seats + baggage (lbf)": max(min(Wf, i.W - We - Wc - float(pay["Weight (lbf)"].sum())), 0.0),
         "Wing weight in table (lbf)": float(items.loc[items["Item"].str.contains("wing", case=False, na=False), "Weight (lbf)"].sum()),
         "Mean chord (in)": c_in, "LEMAC (in)": i.x_LEMAC}
    if len(cases):
        s.update({"Forward CG (in)": cases["CG (in)"].min(), "Aft CG (in)": cases["CG (in)"].max(),
                  "Forward CG (%MAC)": cases["CG (%MAC)"].min(), "Aft CG (%MAC)": cases["CG (%MAC)"].max(),
                  "Min flying weight (lbf)": cases["Weight (lbf)"].min()})
    else:
        s.update({"Forward CG (in)": float("nan"), "Aft CG (in)": float("nan"),
                  "Forward CG (%MAC)": float("nan"), "Aft CG (%MAC)": float("nan"), "Min flying weight (lbf)": float("nan")})
    return s, cases, items


# ---------------------------------------------------------------- stability & control
def stability(i: Inputs, cg_fwd_pct: float, cg_aft_pct: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Longitudinal static stability (Anderson ch. 7) + fuselage term (Raymer) + elevator trim and hinge moments."""
    c, S = chord(i), i.S
    aw = a_wing(i)
    AR_h = i.b_h ** 2 / i.S_h if i.S_h > 0 else 1.0
    at = helmbold(i.a0_tail, AR_h)
    deps = i.deps_override if i.deps_override > 0 else 2 * aw / (math.pi * AR(i))
    VH = i.S_h * i.l_h / (S * c)
    VV = i.S_v * i.l_v / (S * i.b)
    Cma_f = i.Kf * i.w_fus ** 2 * i.L_fus / (c * S) * 57.2958    # per rad
    hn = i.h_ac - Cma_f / aw + i.eta_h * VH * (at / aw) * (1 - deps)
    a_tot = aw + i.eta_h * (i.S_h / S) * at * (1 - deps)
    Cm_de = -i.eta_h * VH * at * i.tau_e                          # per rad
    a0L, it = i.alpha0L * DEG, i.i_t * DEG

    def Cm(alpha_deg, h, de_deg=0.0):
        aa = np.asarray(alpha_deg) * DEG - a0L                    # absolute angle of attack
        alpha_t = (aa + a0L) - it - deps * aa                     # tail AoA (downwash ~ wing lift)
        return (i.Cm_ac + aw * aa * (h - i.h_ac) + Cma_f * aa
                - i.eta_h * VH * at * alpha_t + Cm_de * de_deg * DEG)

    h_f, h_a = cg_fwd_pct / 100, cg_aft_pct / 100
    alpha_stall = (i.CLmax_clean / aw) / DEG + i.alpha0L
    alpha = np.linspace(-6.0, alpha_stall, 120)
    cm_df = pd.DataFrame({"alpha (deg)": alpha, "C_L wing": aw * (alpha - i.alpha0L) * DEG,
                          "Cm fwd CG": Cm(alpha, h_f), "Cm aft CG": Cm(alpha, h_a)})

    def trim_de(alpha_deg, h):
        return -float(Cm(alpha_deg, h)) / (Cm_de * DEG) if Cm_de != 0 else float("nan")

    def alpha_for_CL(CL):
        return CL / aw / DEG + i.alpha0L

    CL_cruise = 2 * i.W / (density(i.altitude) * i.V_cruise ** 2 * S)
    a_cr = alpha_for_CL(float(CL_cruise))
    cm_alpha = lambda h: aw * (h - i.h_ac) + Cma_f - i.eta_h * VH * at * (1 - deps)
    Cm0 = float(Cm(i.alpha0L, h_a))                               # at zero lift
    s = {"Wing 3-D lift slope a_w (1/rad)": aw, "Tail AR": AR_h, "Tail 3-D lift slope a_t (1/rad)": at,
         "d(eps)/d(alpha)": deps, "Horizontal tail volume V_H": VH, "V_H target": i.VH_target,
         "S_h for target V_H (ft^2)": i.VH_target * S * c / i.l_h,
         "Vertical tail volume V_V": VV, "V_V target": i.VV_target,
         "S_v for target V_V (ft^2)": i.VV_target * S * i.b / i.l_v,
         "Fuselage Cm_alpha (1/rad)": Cma_f, "Total lift slope a (1/rad)": a_tot,
         "Neutral point h_n (x/c)": hn, "Neutral point (%MAC)": 100 * hn,
         "Neutral point (in from datum)": i.x_LEMAC + hn * c * 12,
         "Aft CG limit for min SM (%MAC)": 100 * (hn - i.SM_min),
         "Static margin, fwd CG (%MAC)": 100 * (hn - h_f), "Static margin, aft CG (%MAC)": 100 * (hn - h_a),
         "Cm_alpha fwd CG (1/rad)": cm_alpha(h_f), "Cm_alpha aft CG (1/rad)": cm_alpha(h_a),
         "Cm_0 (zero lift, aft CG)": Cm0,
         "Trim alpha (de = 0), aft CG (deg)": i.alpha0L - Cm0 / cm_alpha(h_a) / DEG if cm_alpha(h_a) != 0 else float("nan"),
         "Elevator power Cm_de (1/rad)": Cm_de,
         "Cruise C_L": float(CL_cruise), "Cruise alpha (deg)": a_cr,
         "de trim at cruise, fwd CG (deg)": trim_de(a_cr, h_f), "de trim at cruise, aft CG (deg)": trim_de(a_cr, h_a),
         "de trim at C_L,max clean, fwd CG (deg)": trim_de(alpha_stall, h_f),
         "Stall alpha clean (deg)": alpha_stall}

    # hinge moments over the alpha range for de = -25..+25
    des = sorted({-i.de_max, -15.0, -5.0, 0.0, 5.0, 15.0, i.de_max})
    rows = []
    q_ref = 0.5 * RHO0 * i.V_cruise ** 2
    for a in alpha:
        aa = a * DEG - a0L
        at_deg = ((aa + a0L) - it - deps * aa) / DEG
        for de in des:
            ch = i.Ch0 + i.Ch_alpha * at_deg * DEG + i.Ch_delta * de * DEG
            rows.append({"alpha (deg)": a, "alpha_t (deg)": at_deg, "de (deg)": de, "Ch": ch,
                         "Hinge moment at V_cruise (lbf*ft)": ch * q_ref * i.S_e * i.c_e})
    hinge = pd.DataFrame(rows)
    s["Max |hinge moment| at V_cruise (lbf*ft)"] = float(hinge["Hinge moment at V_cruise (lbf*ft)"].abs().max())
    return s, cm_df, hinge


# ---------------------------------------------------------------- V-n diagram
def vn_diagram(i: Inputs) -> tuple[dict, dict]:
    W, S = i.W, i.S
    WS = W / S
    c = chord(i)
    Vs1 = float(stall_speed(W, RHO0, S, i.CLmax_clean))
    Vs0 = float(stall_speed(W, RHO0, S, i.CLmax_land))
    n1_formula = 2.1 + 24000.0 / (W + 10000.0)
    n1 = min(n1_formula, i.n1_cap)
    n2 = -i.n_neg_ratio * n1
    VA = Vs1 * math.sqrt(n1)
    VC = i.VC_coef * math.sqrt(WS) * KT
    VD = i.VD_ratio * VC if i.VD_basis.startswith("Part 23") else i.VD_cruise_factor * i.V_cruise
    VF = i.VF_ratio * Vs0
    a = a_wing(i)
    mu = 2 * WS / (RHO0 * c * a * G)
    Kg = 0.88 * mu / (5.3 + mu)

    def dn(U, V):
        return Kg * RHO0 * U * V * a / (2 * WS)

    gC, gD = dn(i.Ude_C, VC), dn(i.Ude_D, VD)
    n_pos = max(n1, 1 + gC, 1 + gD)
    n_neg = min(n2, 1 - gC, 1 - gD)
    vals = {"W/S (lbf/ft^2)": WS, "Mean chord c (ft)": c, "Lift slope for gusts (1/rad)": a,
            "V_S1 clean (ft/s)": Vs1, "V_S0 flaps (ft/s)": Vs0,
            "n1 formula 2.1+24000/(W+10000)": n1_formula, "n1 limit": n1, "n2 limit": n2,
            "V_A (ft/s)": VA, "V_A (kt)": VA / KT, "V_C (ft/s)": VC, "V_C (kt)": VC / KT,
            "V_D (ft/s)": VD, "V_D (kt)": VD / KT, "V_F (ft/s)": VF, "V_F (kt)": VF / KT,
            "mu_g": mu, "K_g": Kg,
            "n gust + at V_C": 1 + gC, "n gust - at V_C": 1 - gC,
            "n gust + at V_D": 1 + gD, "n gust - at V_D": 1 - gD,
            "Governing + limit n": n_pos, "Governing - limit n": n_neg,
            "Ultimate + n": i.FoS * n_pos, "Ultimate - n": i.FoS * n_neg}
    V = np.linspace(0, VD, 400)
    q = 0.5 * RHO0 * V ** 2
    pos = np.minimum(q * i.CLmax_clean / WS, n1)
    neg_line = np.where(V <= VC, n2, n2 + (0 - n2) * (V - VC) / max(VD - VC, 1e-9))
    neg = np.maximum(q * i.CL_min / WS, neg_line)
    curves = {"V": V, "pos": pos, "neg": neg,
              "gust_C": (np.array([0, VC]), np.array([1, 1 + gC]), np.array([1, 1 - gC])),
              "gust_D": (np.array([0, VD]), np.array([1, 1 + gD]), np.array([1, 1 - gD]))}
    return vals, curves


# ---------------------------------------------------------------- spar loads
def spar(i: Inputs, n_ult: float) -> tuple[pd.DataFrame, dict]:
    """Schrenk lift distribution on the half wing at ultimate load, shear & moment (cantilever and strut)."""
    half = i.b / 2
    c = chord(i)
    W_wing = i.W_wing_frac * i.W
    L_ult = n_ult * (i.W - W_wing)
    L_half = L_ult / 2
    y = np.linspace(0.0, half, int(i.n_stations))
    if i.strut and 0 < i.y_strut < half:
        y = np.unique(np.append(y, i.y_strut))
    w = 0.5 * (L_half / half + 4 * L_half / (math.pi * half) * np.sqrt(np.clip(1 - (y / half) ** 2, 0, 1)))

    def tip_cumtrapz(f):
        out = np.zeros_like(y)
        for k in range(len(y) - 2, -1, -1):
            out[k] = out[k + 1] + 0.5 * (f[k] + f[k + 1]) * (y[k + 1] - y[k])
        return out

    Vc = tip_cumtrapz(w)
    Mc = tip_cumtrapz(Vc)
    df = pd.DataFrame({"y (ft)": y, "w (lbf/ft)": w, "V cantilever (lbf)": Vc, "M cantilever (lbf*ft)": Mc})
    res = {"W_wing (lbf)": W_wing, "Ultimate n used": n_ult, "L_net,ult (lbf)": L_ult, "L_net,ult/2 (lbf)": L_half,
           "Root shear, cantilever (lbf)": float(Vc[0]), "Root moment, cantilever (lbf*ft)": float(Mc[0])}
    V_use, M_use, y_use = Vc, Mc, y
    if i.strut and 0 < i.y_strut < half:
        ys = i.y_strut
        R = Mc[0] / ys
        inb = y < ys
        Vt = np.where(inb, Vc - R, Vc)
        Mt = np.where(inb, Mc - R * (ys - y), Mc)
        df["V strut-braced (lbf)"] = Vt
        df["M strut-braced (lbf*ft)"] = Mt
        Ls = math.hypot(ys, i.h_strut)
        F = R * Ls / i.h_strut                  # + tension
        H = R * ys / i.h_strut                  # horizontal component -> spar axial (compression for +n)
        res.update({"Strut vertical reaction R_z (lbf)": float(R), "Strut length (ft)": Ls,
                    "Strut axial force (lbf, + tension)": float(F),
                    "Spar axial load inboard of strut (lbf, + compression)": float(H)})
        if F < 0:   # compression -> Euler buckling (pinned-pinned)
            res["Strut EI required for buckling (lbf*in^2)"] = abs(F) * (Ls * 12) ** 2 / math.pi ** 2
            res["Strut I required (in^4)"] = res["Strut EI required for buckling (lbf*in^2)"] / i.E_strut
        V_use, M_use = Vt, Mt
        k = int(np.searchsorted(y, ys))
        if k < len(y) and abs(y[k] - ys) < 1e-9:
            extra = df.iloc[[k]].copy()
            extra["V strut-braced (lbf)"] = Vc[k] - R
            df = pd.concat([df.iloc[:k], extra, df.iloc[k:]], ignore_index=True)
            V_use, M_use, y_use = np.append(V_use, Vc[k] - R), np.append(M_use, Mt[k]), np.append(y, ys)
    kM = int(np.argmax(np.abs(M_use)))
    Vmax, Mmax = float(np.max(np.abs(V_use))), float(np.abs(M_use[kM]))
    res["Design max |V| (lbf)"] = Vmax
    res["Design max |M| (lbf*ft)"] = Mmax
    res["Design max |M| at y (ft)"] = float(y_use[kM])
    for name, xc, share, tc in [("front", i.x_front, i.share_front, i.tc_front),
                                ("rear", i.x_rear, i.share_rear, i.tc_rear)]:
        h = tc * c * 12 - 2 * i.clearance       # in
        res[f"{name} spar x (ft)"] = xc * c
        res[f"{name} spar depth h (in)"] = h
        res[f"{name} spar design M (lbf*ft)"] = share * Mmax
        res[f"{name} spar design V (lbf)"] = share * Vmax
        res[f"{name} spar cap area req. (in^2)"] = share * Mmax * 12 / (i.sigma_allow * h) if h > 0 else float("nan")
        res[f"{name} spar web thickness req. (in)"] = share * Vmax / (i.tau_allow * h) if h > 0 else float("nan")
    res["Spar spacing (ft)"] = (i.x_rear - i.x_front) * c
    return df, res


# ---------------------------------------------------------------- run everything
@dataclass
class Results:
    inputs: Inputs
    summary: dict[str, Any]
    power: pd.DataFrame
    climb: pd.DataFrame
    takeoff: dict
    landing: dict
    gradients: dict
    cruise: dict
    power_settings: pd.DataFrame
    turn: pd.DataFrame
    turn_summary: dict
    glide: pd.DataFrame
    glide_summary: dict
    weights: dict
    loading_cases: pd.DataFrame
    weight_items: pd.DataFrame
    stab: dict
    cm_alpha: pd.DataFrame
    hinge: pd.DataFrame
    vn: dict
    vn_curves: dict
    spar: pd.DataFrame
    spar_summary: dict
    spar_neg: pd.DataFrame
    spar_neg_summary: dict
    propulsion: pd.DataFrame
    checks: pd.DataFrame
    warnings: list[str]


def _core(i: Inputs) -> dict:
    """Everything that depends on weight (used twice: MTOW and comparison weight)."""
    rho = float(density(i.altitude))
    pw = power_table(i, rho=rho)
    climb, ceil = climb_table(i)
    rc_max, v_rc = max_climb(i, rho)
    to = {"no flaps": takeoff(i, i.CLmax_clean, False), "flaps": takeoff(i, i.CLmax_TO, True)}
    ld = {"no flaps": landing(i, i.CLmax_clean, False), "flaps": landing(i, i.CLmax_land, True)}
    grads = {"Takeoff (TO flaps)": climb_gradient(i, i.CD0 + i.dCD0_flaps_TO, i.CLmax_TO),
             "Balked landing (landing flaps)": climb_gradient(i, i.CD0 + i.dCD0_flaps, i.CLmax_land)}
    Vs = float(stall_speed(i.W, rho, i.S, i.CLmax_clean))
    summary = {
        "Weight (lbf)": i.W, "Aspect ratio AR": AR(i), "K = 1/(pi e AR)": K_ind(i),
        "Wing loading W/S (lbf/ft^2)": i.W / i.S, "Power loading W/P (lbf/hp)": i.W / i.P_max,
        "Density at altitude (slug/ft^3)": rho,
        "V_stall clean (ft/s)": Vs, "V_stall clean (kt)": Vs / KT,
        "V_stall TO flaps (kt)": float(stall_speed(i.W, rho, i.S, i.CLmax_TO)) / KT,
        "V_stall landing flaps (kt)": float(stall_speed(i.W, rho, i.S, i.CLmax_land)) / KT,
        "V_LO (ft/s)": i.VLO_factor * Vs,
        "(L/D)max": 1.0 / math.sqrt(4 * i.CD0 * K_ind(i)),
        "V_max level (ft/s)": _vmax_from_table(pw), "V_max level (kt)": _vmax_from_table(pw) / KT,
        "Max R/C (ft/min)": rc_max * 60, "V for max R/C (kt)": v_rc / KT,
        "Service ceiling (ft)": ceil["service_ceiling_ft"], "Absolute ceiling (ft)": ceil["absolute_ceiling_ft"],
        "Runway limit 25b (ft)": i.runway_factor * i.b,
        "Takeoff roll, no flaps (ft)": to["no flaps"]["Ground roll (ft)"],
        "Takeoff roll, flaps (ft)": to["flaps"]["Ground roll (ft)"],
        "Landing roll, no flaps (ft)": ld["no flaps"]["Ground roll (ft)"],
        "Landing roll, flaps (ft)": ld["flaps"]["Ground roll (ft)"],
        "Takeoff climb gradient (%)": grads["Takeoff (TO flaps)"]["Max gradient (%)"],
        "Balked landing gradient (%)": grads["Balked landing (landing flaps)"]["Max gradient (%)"],
    }
    return dict(pw=pw, climb=climb, to=to, ld=ld, grads=grads, summary=summary)


def run(i: Inputs | None = None) -> Results:
    i = i or Inputs()
    core = _core(i)
    summary = core["summary"]

    wsum, cases, items = weights_cg(i)
    W_fuel = wsum["Full fuel (lbf)"]
    cruise, ps = range_endurance(i, W_fuel)
    turn, turn_s = turn_table(i)
    gl, gl_s = glide(i)
    stab, cm_df, hinge = stability(i, wsum["Forward CG (%MAC)"], wsum["Aft CG (%MAC)"])
    vn, curves = vn_diagram(i)
    sp, sp_s = spar(i, vn["Ultimate + n"])
    spn, spn_s = spar(i, vn["Ultimate - n"])

    summary.update({
        "Max range (nmi)": cruise["Max range (nmi)"], "Max endurance (h)": cruise["Max endurance (h)"],
        "Empty weight (lbf)": wsum["Empty weight (lbf)"], "Useful load (lbf)": wsum["Useful load (lbf)"],
        "CG range (%MAC)": f"{wsum['Forward CG (%MAC)']:.1f} - {wsum['Aft CG (%MAC)']:.1f}",
        "Neutral point (%MAC)": stab["Neutral point (%MAC)"],
        "Static margin, aft CG (%MAC)": stab["Static margin, aft CG (%MAC)"],
        "V_A (kt)": vn["V_A (kt)"], "V_C (kt)": vn["V_C (kt)"], "V_D (kt)": vn["V_D (kt)"],
        "Ultimate load factor +": vn["Ultimate + n"], "Ultimate load factor -": vn["Ultimate - n"],
        "Spar design max moment (lbf*ft)": sp_s["Design max |M| (lbf*ft)"],
    })

    # propulsion (engine + prop combined)
    Vp = np.linspace(0.0, i.V_hi, 141)
    prop = pd.DataFrame({"V (ft/s)": Vp, "V (kt)": Vp / KT, "J": advance_ratio(i, Vp),
                         "eta_P": prop_eta(i, advance_ratio(i, Vp))})
    for h in (0.0, 5000.0, 10000.0):
        rho = float(density(h))
        prop[f"Thrust {h:,.0f} ft (lbf)"] = thrust(i, Vp, rho)
        prop[f"P_A {h:,.0f} ft (hp)"] = power_available_hp(i, Vp, rho)

    # requirement checks
    runway = i.runway_factor * i.b
    g_to = core["grads"]["Takeoff (TO flaps)"]["Max gradient (%)"]
    g_bl = core["grads"]["Balked landing (landing flaps)"]["Max gradient (%)"]
    vmax = summary["V_max level (ft/s)"]
    M_lim = i.mach_max
    chk = []
    if i.runway_factor > 0:
        for lab, key in [("Takeoff ground roll, no flaps (ft)", "Takeoff roll, no flaps (ft)"),
                         ("Takeoff ground roll, flaps (ft)", "Takeoff roll, flaps (ft)"),
                         ("Landing ground roll, no flaps (ft)", "Landing roll, no flaps (ft)"),
                         ("Landing ground roll, flaps (ft)", "Landing roll, flaps (ft)")]:
            req = f"<= {runway:.0f} ft ({i.runway_factor:g} x span)"
            chk.append(("Design requirement", lab, summary[key], req, summary[key] <= runway))
    chk += [
        ("Part 23 §23.2120(a)", "Takeoff climb gradient (%)", g_to, f">= {100*i.grad_TO_req:.1f} %", g_to >= 100 * i.grad_TO_req),
        ("Part 23 §23.2120(c)", "Balked landing gradient (%)", g_bl, f">= {100*i.grad_balked_req:.1f} %", g_bl >= 100 * i.grad_balked_req),
        ("Design requirement", "Mach at V_D (-)", vn["V_D (ft/s)"] / A0, f"<= {M_lim:.2f}", vn["V_D (ft/s)"] / A0 <= M_lim),
        ("Design requirement", "Mach at level V_max (-)", vmax / A0, f"<= {M_lim:.2f}", (vmax / A0 <= M_lim) if vmax == vmax else False),
        ("Stability", "Static margin at aft CG (%MAC)", stab["Static margin, aft CG (%MAC)"], f">= {100*i.SM_min:.0f} %", stab["Static margin, aft CG (%MAC)"] >= 100 * i.SM_min),
        ("Stability", "Cm_alpha at aft CG (1/rad)", stab["Cm_alpha aft CG (1/rad)"], "< 0", stab["Cm_alpha aft CG (1/rad)"] < 0),
        ("Control", "Elevator to trim C_L,max at fwd CG (deg)", stab["de trim at C_L,max clean, fwd CG (deg)"], f"|de| <= {i.de_max:.0f}", abs(stab["de trim at C_L,max clean, fwd CG (deg)"]) <= i.de_max),
        ("Guideline", "Horizontal tail volume V_H", stab["Horizontal tail volume V_H"], f"~ {i.VH_target:.2f} (Raymer)", stab["Horizontal tail volume V_H"] >= 0.85 * i.VH_target),
        ("Guideline", "Vertical tail volume V_V", stab["Vertical tail volume V_V"], f"~ {i.VV_target:.3f} (Raymer)", stab["Vertical tail volume V_V"] >= 0.85 * i.VV_target),
        ("Weights", "Payload with full fuel (lbf)", wsum["Payload with full fuel (lbf)"], ">= 0", wsum["Payload with full fuel (lbf)"] >= 0),
    ]
    checks = pd.DataFrame(chk, columns=["Source", "Check", "Value", "Requirement", "Pass"])

    warn = []
    if i.eta_mode == "table" and i.eta_J == DEFAULT_ETA_J and i.eta_vals == DEFAULT_ETA:
        warn.append("The propeller efficiency curve is a generic placeholder. Enter data for your propeller.")
    if i.strut and not (0 < i.y_strut < i.b / 2):
        warn.append("Strut location must be between root and tip.")
    if W_fuel <= 0:
        warn.append("No 'fuel' rows in the weight table: range/endurance can't be computed.")
    if wsum["Empty weight (lbf)"] >= i.W:
        warn.append("Empty weight from the weight table is >= MTOW.")
    ww = wsum["Wing weight in table (lbf)"]
    if ww > 0 and abs(ww - i.W_wing_frac * i.W) > 0.1 * ww:
        warn.append(f"Wing weight in table ({ww:.0f} lbf) differs from W_wing_frac x W ({i.W_wing_frac*i.W:.0f} lbf) used by the spar.")
    for _, r in checks[~checks["Pass"]].iterrows():
        v = r["Value"]
        warn.append(f"FAILS: {r['Check']} = {v:,.3g} (needs {r['Requirement']})")

    return Results(i, summary, core["pw"], core["climb"], core["to"], core["ld"], core["grads"], cruise, ps,
                   turn, turn_s, gl, gl_s, wsum, cases, items, stab, cm_df, hinge, vn, curves,
                   sp, sp_s, spn, spn_s, prop, checks, warn)


# ---------------------------------------------------------------- export
def to_excel(res: Results, path) -> Any:
    """Dump every input and table to an .xlsx ."""
    def kv(d):
        return pd.DataFrame(list(d.items()), columns=["Quantity", "Value"])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        d = asdict(res.inputs)
        d.pop("weight_items")
        d["eta_J"] = ", ".join(map(str, d["eta_J"])); d["eta_vals"] = ", ".join(map(str, d["eta_vals"]))
        pd.DataFrame(list(d.items()), columns=["Input", "Value"]).to_excel(xw, sheet_name="Inputs", index=False)
        res.checks.to_excel(xw, sheet_name="Requirement checks", index=False)
        kv(res.summary).to_excel(xw, sheet_name="Summary", index=False)
        res.weight_items.to_excel(xw, sheet_name="Weight build-up", index=False)
        kv(res.weights).to_excel(xw, sheet_name="Weights & CG", index=False)
        res.loading_cases.to_excel(xw, sheet_name="Loading cases", index=False)
        kv(res.stab).to_excel(xw, sheet_name="Stability", index=False)
        res.cm_alpha.to_excel(xw, sheet_name="Cm vs alpha", index=False)
        res.hinge.to_excel(xw, sheet_name="Hinge moments", index=False)
        res.power.to_excel(xw, sheet_name="Power curves", index=False)
        res.propulsion.to_excel(xw, sheet_name="Propulsion", index=False)
        res.climb.to_excel(xw, sheet_name="Climb & ceiling", index=False)
        pd.DataFrame(res.gradients).to_excel(xw, sheet_name="Climb gradients")
        pd.DataFrame(res.takeoff).to_excel(xw, sheet_name="Takeoff")
        pd.DataFrame(res.landing).to_excel(xw, sheet_name="Landing")
        kv(res.cruise).to_excel(xw, sheet_name="Cruise", index=False)
        res.power_settings.to_excel(xw, sheet_name="Range vs power", index=False)
        res.turn.to_excel(xw, sheet_name="Turning", index=False)
        res.glide.to_excel(xw, sheet_name="Glide", index=False)
        kv(res.vn).to_excel(xw, sheet_name="V-n", index=False)
        res.spar.to_excel(xw, sheet_name="Spar loads +n", index=False)
        res.spar_neg.to_excel(xw, sheet_name="Spar loads -n", index=False)
        pd.DataFrame({"+n case": pd.Series(res.spar_summary), "-n case": pd.Series(res.spar_neg_summary)}
                     ).to_excel(xw, sheet_name="Spar summary")
    return path


if __name__ == "__main__":
    r = run()
    w = max(len(k) for k in r.summary)
    for k, v in r.summary.items():
        print(f"{k:<{w}}  {v:,.4g}" if isinstance(v, float) else f"{k:<{w}}  {v}")
    print()
    print(r.checks.to_string(index=False))
    for m in r.warnings:
        print("WARNING:", m)
