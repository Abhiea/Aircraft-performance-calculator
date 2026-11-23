"""
Interactive front end for the calculator.   Run with:   streamlit run app.py
Pages: Calculator, How to use, Equations.
Inputs are generated from core.Inputs. The core works in imperial units; units.py converts for display.
"""
from __future__ import annotations

import importlib
import io
import json
import zipfile
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core
import docs_pages
import labels
import plots
import units

# Streamlit's file watcher can miss edits in OneDrive folders: always use the latest modules.
for _m in (core, units, labels, plots, docs_pages):
    importlib.reload(_m)
tex = labels.tex

st.set_page_config(page_title="Aircraft Performance Calculator", layout="wide")

# Loading indicator: replace Streamlit's running figure with a plain rotating circle, in the same slot,
# so it lines up with Stop / Deploy / the menu.
st.markdown("""
<style>
[data-testid="stStatusWidgetRunningManIcon"] { display: none !important; }
[data-testid="stStatusWidgetRunningIcon"] {
    display: flex !important; align-items: center; justify-content: center;
    width: 1.6rem; height: 1.6rem;
}
[data-testid="stStatusWidgetRunningIcon"]::after {
    content: ""; box-sizing: border-box;
    width: 1.05rem; height: 1.05rem;
    border: 2px solid rgba(160, 160, 160, 0.35);
    border-top-color: rgba(250, 250, 250, 0.85);
    border-radius: 50%;
    animation: calc-spin 0.8s linear infinite;
}
@keyframes calc-spin { to { transform: rotate(360deg); } }
div[data-testid="stMarkdownContainer"] table { font-size: 0.88rem; }
div[data-testid="stMarkdownContainer"] td { white-space: nowrap; }
div[data-testid="stMarkdownContainer"] th { font-weight: 500 !important; }
h1 { font-weight: 600 !important; font-size: 2.1rem !important; }
h2, h3, h4, h5, h6 { font-weight: 500 !important; }
[data-testid="stMetricValue"] { font-weight: 400 !important; font-size: 1.9rem !important; }
[data-testid="stMetricLabel"] p { font-size: 0.85rem !important; }
[data-testid="stExpander"] summary p { font-weight: 400 !important; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.15rem 0; }
</style>
""", unsafe_allow_html=True)

FIELDS = core.input_fields()
DEFAULTS = asdict(core.Inputs())
TYPES = ["empty", "crew", "payload", "fuel"]
WT_COLS = {"Weight (lbf)": "lbf", "Arm (in)": "in"}


# ------------------------------------------------------------------ state (canonical values are imperial)
def _set_inputs(d: dict):
    for name, _, _ in FIELDS:
        if name in d:
            st.session_state[f"cv_{name}"] = d[name]
        for s in units.SYSTEMS + [""]:
            st.session_state.pop(f"in_{name}_{s}", None)
    st.session_state["eta_df"] = pd.DataFrame({"J": d.get("eta_J", DEFAULTS["eta_J"]),
                                               "eta": d.get("eta_vals", DEFAULTS["eta_vals"])})
    st.session_state["wt_canon"] = pd.DataFrame(d.get("weight_items", DEFAULTS["weight_items"]))
    _reset_editors()


def _reset_editors():
    for k in list(st.session_state.keys()):
        if k.startswith(("wt_editor", "wt_base", "eta_editor")):
            del st.session_state[k]


if "cv_W" not in st.session_state:
    _set_inputs(DEFAULTS)
    st.session_state["baseline"] = None
    st.session_state["units_pref"] = units.IMPERIAL


@st.cache_data(show_spinner=False)
def compute(inputs_json: str) -> core.Results:
    return core.run(core.Inputs.from_dict(json.loads(inputs_json)))


# ------------------------------------------------------------------ formatting helpers
def num(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if v != v:
            return "n/a"
        if v in (float("inf"), float("-inf")):
            return "not reached"
        a = abs(v)
        if a >= 1e7:
            return f"{v:.4g}"
        if a >= 1000:
            return f"{v:,.0f}"
        return "0" if a == 0 else f"{v:.4g}"
    return tex(str(v))


def md_table(df: pd.DataFrame, index: bool = False, cell=num) -> str:
    d = df.copy()
    if index:
        d = d.reset_index().rename(columns={"index": ""})
    cols = list(d.columns)
    align = [("--:" if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]) else ":--")
             for c in cols]
    out = ["| " + " | ".join(tex(str(c)) for c in cols) + " |", "|" + "|".join(align) + "|"]
    for row in d.itertuples(index=False):
        cells = []
        for v in row:
            c = cell(v)
            cells.append(c if "$" in c else c.replace("|", r"\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def page_parts():
    """Helpers bound to the current unit system."""
    sys_ = st.session_state["units_pref"]

    def show_table(df, height=None, index=False, name=None, rows=False):
        disp = units.convert_rows(df, sys_) if rows else units.convert_columns(df, sys_)
        md = md_table(disp, index=index or rows)
        if height:
            with st.container(height=height):
                st.markdown(md)
        else:
            st.markdown(md)
        if name:
            st.download_button(f"Download {name} (.csv)", disp.to_csv(index=index or rows), f"{name}.csv",
                               "text/csv", key=f"dl_{name}")

    def show_kv(d, height=None):
        dd = units.convert_dict(d, sys_)
        show_table(pd.DataFrame(list(dd.items()), columns=["Quantity", "Value"]), height=height)

    return sys_, show_table, show_kv


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ================================================================== calculator page
def calculator():
    # ---------------- sidebar
    if "unit_radio" not in st.session_state:
        st.session_state["unit_radio"] = st.session_state["units_pref"]
    sys_ = st.sidebar.radio("Units", units.SYSTEMS, key="unit_radio", horizontal=True)
    st.session_state["units_pref"] = sys_
    if st.session_state.get("_last_sys") != sys_:
        _reset_editors()
        st.session_state["_last_sys"] = sys_
    plots.set_units(sys_)

    st.sidebar.title("Inputs")
    c1 = c2 = st.sidebar
    if c1.button("Reset to Cessna 172 values", width="stretch"):
        _set_inputs(DEFAULTS)
        st.rerun()
    pin_slot = st.sidebar.empty()
    up = st.sidebar.file_uploader("Load inputs (.json)", type="json")
    if up is not None and st.session_state.get("_loaded") != up.name + str(up.size):
        _set_inputs(json.load(up))
        st.session_state["_loaded"] = up.name + str(up.size)
        st.rerun()

    groups: dict[str, list] = {}
    for name, meta, default in FIELDS:
        groups.setdefault(meta["group"], []).append((name, meta, default))

    values = {}
    for gname, items in groups.items():
        with st.sidebar.expander(tex(gname), expanded=gname in ("Weights & CG", "Wing geometry")):
            if gname == "Weights & CG":
                st.markdown("Component weights " + docs_pages.BADGE["placeholder"])
                st.caption("Distances are measured aft of the datum. Type: empty, crew (always aboard), "
                           "payload (optional), fuel (zero to full).")
                if f"wt_base_{sys_}" not in st.session_state:
                    base = st.session_state["wt_canon"].copy()
                    ren = {}
                    for col, u in WT_COLS.items():
                        nu, f = units.unit_factor(u, sys_)
                        base[col] = pd.to_numeric(base[col], errors="coerce") * f
                        ren[col] = col.replace(f"({u})", f"({nu})")
                    st.session_state[f"wt_base_{sys_}"] = (base.rename(columns=ren), ren)
                base, ren = st.session_state[f"wt_base_{sys_}"]
                wt = st.data_editor(base, num_rows="dynamic", key=f"wt_editor_{sys_}", width="stretch",
                                    hide_index=True, column_config={
                                        "Type": st.column_config.SelectboxColumn("Type", options=TYPES, required=True)})
                inv = {v: k for k, v in ren.items()}
                canon = wt.rename(columns=inv).copy()
                for col, u in WT_COLS.items():
                    canon[col] = pd.to_numeric(canon[col], errors="coerce") / units.unit_factor(u, sys_)[1]
                canon = canon.dropna(subset=["Weight (lbf)"])
                st.session_state["wt_canon"] = canon
                values["weight_items"] = canon.to_dict("records")
            for name, meta, default in items:
                unit = meta["unit"]
                hp = tex(meta["desc"] + (f"  \n\nDefault: {meta['help']}" if meta["help"] else ""))
                tag = docs_pages.badges(meta)
                if meta["options"] or isinstance(default, bool):
                    key = f"in_{name}_"
                    if meta["options"] and st.session_state.get(f"cv_{name}") not in meta["options"]:
                        st.session_state[f"cv_{name}"] = default
                    if key not in st.session_state:
                        st.session_state[key] = st.session_state[f"cv_{name}"]
                    lab = tex(meta["label"]) + (" " + tag if tag else "")
                    v = (st.selectbox(lab, meta["options"], key=key, help=hp) if meta["options"]
                         else st.checkbox(lab, key=key, help=hp))
                    st.session_state[f"cv_{name}"] = values[name] = v
                    continue
                du, f = units.unit_factor(unit, sys_)
                label = tex(meta["label"] + (f" [{du}]" if du and du != "-" else "")) + (" " + tag if tag else "")
                key = f"in_{name}_{sys_}"
                if key not in st.session_state:
                    cv = st.session_state[f"cv_{name}"]
                    st.session_state[key] = int(cv) if isinstance(default, int) else float(f"{cv * f:.6g}")
                    st.session_state[f"shown_{key}"] = st.session_state[key]
                if isinstance(default, int):
                    v = int(st.number_input(label, key=key, step=int(meta["step"] or 1),
                                            min_value=meta["lo"], max_value=meta["hi"], help=hp))
                    cv = v
                else:
                    step = float(f"{float(meta['step'] or 0.01) * f:.2g}")
                    mag = abs(st.session_state[key]) or 1.0
                    fmt_ = ("%.0f" if step >= 1 and mag >= 100 else "%.2f" if step >= 0.01 else
                            "%.3f" if step >= 0.001 else "%.4f")
                    v = st.number_input(label, key=key, step=step, format=fmt_,
                                        min_value=None if meta["lo"] is None else float(meta["lo"]) * f,
                                        max_value=None if meta["hi"] is None else float(meta["hi"]) * f, help=hp)
                    # only write back when the user changed the number, so display rounding never drifts the value
                    cv = st.session_state[f"cv_{name}"] if v == st.session_state.get(f"shown_{key}") else v / f
                st.session_state[f"cv_{name}"] = values[name] = cv
            if gname == "Propulsion":
                st.markdown(tex("Propeller efficiency eta against advance ratio J ") + docs_pages.BADGE["placeholder"])
                st.caption("Used when the model is 'table'. Replace with your propeller's data.")
                eta_df = st.data_editor(st.session_state["eta_df"], num_rows="dynamic", key="eta_editor", width="stretch")
                eta_df = eta_df.dropna().sort_values("J")
                values["eta_J"] = [float(x) for x in eta_df["J"]]
                values["eta_vals"] = [float(x) for x in eta_df["eta"]]

    inputs = core.Inputs.from_dict(values)
    inputs_json = inputs.to_json()
    res = compute(inputs_json)
    I = res.inputs

    st.sidebar.download_button("Save inputs (.json)", inputs_json, "aircraft_inputs.json", "application/json",
                               width="stretch")
    if pin_slot.button("Pin as baseline", width="stretch", help="Headline metrics then show the change from the pinned design"):
        st.session_state["baseline"] = dict(res.summary)
        st.rerun()

    _, show_table, show_kv = page_parts()
    cv = lambda v, u: v * units.unit_factor(u, sys_)[1]
    uu = lambda u: tex(f"({units.unit_factor(u, sys_)[0]})")[1:-1]

    # ---------------- header
    st.session_state["last_inputs_json"] = inputs_json
    st.title("Aircraft Performance Calculator")
    st.caption("Performance, stability and wing loads of a light propeller aircraft")
    n_chk, n_fail = len(res.checks), int((~res.checks["Pass"]).sum())
    status = (f"All {n_chk} requirement checks pass." if n_fail == 0 else
              f"{n_chk - n_fail} of {n_chk} requirement checks pass. :red[{n_fail} failing], see Overview.")
    ph = [n for n, mt, d in FIELDS if mt["status"] == "placeholder" and st.session_state.get(f"cv_{n}") == d]
    if I.eta_mode == "table" and I.eta_J == core.DEFAULT_ETA_J and I.eta_vals == core.DEFAULT_ETA:
        ph.append("eta_curve")
    xf = [n for n, mt, d in FIELDS if mt["xflr5"]]
    with st.container(border=True):
        st.markdown(status + f"  \n{len(ph)} inputs still use placeholder values and {len(xf)} should come from "
                    "XFLR5 for a new design. Both are marked in the sidebar.")
    notes = [w for w in res.warnings if not w.startswith("FAILS")]
    if notes:
        with st.expander(f"Notes ({len(notes)})"):
            for w in notes:
                st.markdown("- " + tex(w))

    base = st.session_state["baseline"]

    def metric(col, label, key, f="{:,.0f}", invert=False):
        lab, fac = units.convert_label(label, sys_)
        val = res.summary[key]
        delta = None
        if base and key in base and pd.notna(base[key]) and pd.notna(val):
            d = (val - base[key]) * fac
            delta = f.format(d) if abs(d) > 1e-12 else None
        col.metric(tex(lab), "n/a" if pd.isna(val) else f.format(val * fac), delta,
                   delta_color="inverse" if invert else "normal", border=True)

    m = st.columns(3)
    metric(m[0], "Stall speed (kt)", "V_stall clean (kt)", "{:.1f}", invert=True)
    metric(m[1], "Top speed (kt)", "V_max level (kt)", "{:.1f}")
    metric(m[2], "Max climb (ft/min)", "Max R/C (ft/min)", "{:,.0f}" if sys_ == units.IMPERIAL else "{:.2f}")
    m = st.columns(3)
    metric(m[0], "Takeoff roll (ft)", "Takeoff roll, flaps (ft)", invert=True)
    metric(m[1], "Range (nmi)", "Max range (nmi)")
    metric(m[2], "Static margin (%)", "Static margin, aft CG (%MAC)", "{:.1f}")
    if base:
        st.caption("Small numbers under each value show the change from the pinned baseline.")

    def sub(t):
        st.markdown(f"#### {t}")

    tabs = st.tabs(["Overview", "Performance", "Weight and balance", "Stability and control",
                    "Loads and structure", "Export"])

    # ---------------- Overview
    with tabs[0]:
        sub("Requirement checks")
        chk = units.convert_rows(res.checks, sys_, label_col="Check")
        chk["Value"] = chk["Value"].map(num)
        chk["Requirement"] = chk["Requirement"].map(lambda t: units.convert_text(t, sys_))
        chk["Result"] = chk.pop("Pass").map(lambda p: "Pass" if p else ":red[Fail]")
        chk["Source"] = chk["Source"].str.replace("§", "§ ", regex=False)
        st.markdown(md_table(chk, cell=lambda v: v if isinstance(v, str) and v.startswith(":red") else num(v)))
        st.caption(r"Part 23 checks use sea-level standard conditions. Guideline rows are typical values (Raymer), "
                   r"not rules. Lateral and directional stability ($C_{l_\beta}$, $C_{n_\beta}$) is not calculated here.")
        sub("Key results")
        show_kv(res.summary, height=640)
        if base:
            sub("Change from baseline")
            keys = [k for k in res.summary if isinstance(res.summary[k], (int, float)) and k in base
                    and isinstance(base[k], (int, float))]
            comp = pd.DataFrame({"Baseline": [base[k] for k in keys], "Current": [res.summary[k] for k in keys]},
                                index=keys)
            comp["Change"] = comp["Current"] - comp["Baseline"]
            comp["Change (%)"] = 100 * comp["Change"] / comp["Baseline"].abs()
            show_table(comp, height=420, rows=True)

    # ---------------- Performance
    with tabs[1]:
        sections = ["Power and climb", "Propulsion", "Takeoff and landing", "Cruise and range", "Turn and glide"]
        sec = st.segmented_control("Section", sections, default=sections[0], key="perf_section",
                                   label_visibility="collapsed") or sections[0]
        if sec == "Power and climb":
            opts = [0, 2500, 5000, 7500, 10000, 12500, 15000]
            alts = st.multiselect(f"Altitudes to compare ({units.unit_factor('ft', sys_)[0]})", opts,
                                  default=[int(I.altitude)] if I.altitude in opts else [0],
                                  format_func=lambda h: f"{cv(h, 'ft'):,.0f}")
            a, b = st.columns(2)
            with a:
                show(plots.power_curves(res, alts or [I.altitude]))
            with b:
                show(plots.rate_of_climb(res))
            sub("Part 23 climb gradients (sea level, full power)")
            show_table(pd.DataFrame(res.gradients).T, index=True)
            sub("Climb, ceiling and time to climb")
            a, b = st.columns(2)
            with a:
                show(plots.climb_vs_altitude(res))
            with b:
                show_table(res.climb, height=480, name="climb")
            with st.expander(f"Full power table at {cv(I.altitude, 'ft'):,.0f} {uu('ft')}"):
                st.caption(tex("Rows where C_L > C_L,max are marked stalled and ignored for top speed and climb."))
                show_table(res.power, height=420, name="power_curves")
        elif sec == "Propulsion":
            show(plots.propulsion(res))
            with st.expander("Propulsion table"):
                show_table(res.propulsion, height=420, name="propulsion")
        elif sec == "Takeoff and landing":
            if I.runway_factor > 0:
                rw = cv(res.summary["Runway limit 25b (ft)"], "ft")
                st.markdown(f"Ground roll limit: {I.runway_factor:g} $\\times$ span = {rw:,.0f} {uu('ft')}.")
            a, b = st.columns(2)
            with a:
                sub("Takeoff")
                show_table(pd.DataFrame(res.takeoff), rows=True)
            with b:
                sub("Landing")
                show_table(pd.DataFrame(res.landing), rows=True)
            st.caption(tex(f"C_L,max used: takeoff flaps {I.CLmax_TO:.2f}, landing flaps {I.CLmax_land:.2f}, "
                           f"clean {I.CLmax_clean:.2f}. Ground rolls are without an obstacle."))
        elif sec == "Cruise and range":
            a, b = st.columns(2)
            with a:
                sub("Cruise, range and endurance")
                show_kv(res.cruise)
            with b:
                show(plots.range_vs_power(res))
            sub("Range and endurance against power setting")
            show_table(res.power_settings, name="range_vs_power")
            show(plots.drag_polar(res))
        else:
            a, b = st.columns(2)
            with a:
                sub("Level turn")
                show_kv(res.turn_summary)
                show(plots.turning(res))
            with b:
                sub("Glide")
                gs = pd.DataFrame(res.glide_summary).rename(columns={"clean": "Clean", "flaps": "Landing flaps"})
                show_table(gs, rows=True)
                show(plots.hodograph(res))
            with st.expander("Turn table"):
                show_table(res.turn, height=400, name="turning")
            with st.expander("Glide table"):
                show_table(res.glide, height=400, name="glide")

    # ---------------- Weight and balance
    with tabs[2]:
        a, b = st.columns([2, 3])
        with a:
            sub("Weight limits and CG range")
            show_kv(res.weights)
        with b:
            show(plots.cg_envelope(res))
        with st.expander("Weight build-up"):
            show_table(res.weight_items, name="weight_buildup")
        with st.expander("All loading cases"):
            st.caption("Every payload combination, with fuel at zero and at the most allowed by MTOW.")
            show_table(res.loading_cases.sort_values("CG (in)"), height=400, name="loading_cases")

    # ---------------- Stability and control
    with tabs[3]:
        a, b = st.columns([2, 3])
        with a:
            sub("Longitudinal stability and trim")
            show_kv(res.stab, height=820)
        with b:
            show(plots.cm_alpha(res))
            show(plots.hinge(res))
        with st.expander(tex("Cm vs alpha table")):
            show_table(res.cm_alpha, height=400, name="cm_alpha")
        with st.expander("Hinge moment table"):
            show_table(res.hinge, height=400, name="hinge_moments")

    # ---------------- Loads and structure
    with tabs[4]:
        sub("V-n diagram")
        a, b = st.columns([3, 2])
        with a:
            show(plots.vn(res))
        with b:
            show_kv(res.vn, height=560)
        sub("Wing spar")
        a, b = st.columns([3, 2])
        with a:
            show(plots.spar_loads(res))
        with b:
            sp = pd.DataFrame({"+n case": pd.Series(res.spar_summary), "-n case": pd.Series(res.spar_neg_summary)})
            show_table(sp, rows=True, height=760)
        st.caption("Loads use the Schrenk distribution at the ultimate load factors. With a strut, the wing root is "
                   "pinned and the strut carries the root moment. The strut also pushes axially on the spar inboard of "
                   "the strut, which should be combined with bending when sizing the caps. In the $-n$ case the strut is "
                   "in compression, so it is checked for buckling.")
        with st.expander(tex("Spar table, +n case")):
            show_table(res.spar, height=400, name="spar_loads_pos")
        with st.expander(tex("Spar table, -n case")):
            show_table(res.spar_neg, height=400, name="spar_loads_neg")

    # ---------------- Export
    with tabs[5]:
        st.markdown("The Excel file always uses imperial units. Figures use the selected unit system.")
        buf = io.BytesIO()
        core.to_excel(res, buf)
        st.download_button("All tables (.xlsx)", buf.getvalue(), "aircraft_results.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button("Prepare all figures (.zip)"):
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as z:
                for nm, fn in plots.ALL.items():
                    f = fn(res)
                    ib = io.BytesIO()
                    f.savefig(ib, format="png", dpi=200, bbox_inches="tight")
                    plt.close(f)
                    z.writestr(f"{nm}.png", ib.getvalue())
            st.download_button("Download figures (.zip)", zbuf.getvalue(), "aircraft_figures.zip", "application/zip")
        st.download_button("Inputs (.json)", inputs_json, "aircraft_inputs.json", "application/json", key="dl_json2")


# ================================================================== navigation
pg = st.navigation([
    st.Page(calculator, title="Calculator", default=True),
    st.Page(docs_pages.how_to, title="How to use", url_path="how-to-use"),
    st.Page(docs_pages.equations, title="Equations", url_path="equations"),
])
pg.run()
