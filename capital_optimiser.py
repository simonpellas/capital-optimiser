# capital_optimiser_v5.py
# -----------------------------------------------------------------------------
# How to run
# 1) python -m pip install streamlit pulp pandas
# 2) python -m streamlit run capital_optimiser_v5.py
# -----------------------------------------------------------------------------

import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
import pulp


# ------------------------------- Helpers ------------------------------------ #

RATE_LOW = 0.148   # 14.8% CET1 per RWA up to threshold
RATE_HIGH = 0.173  # 17.3% CET1 per RWA above threshold

# UPDATED DEFAULTS (per your screenshot)
DEFAULT_JSON = """{
  "capital": {"CET1": 126.45, "leverage_ratio": 0.0350, "RWA_threshold": 588},
  "assets": [
    {"name":"Lombard","nim":0.015,"rw":0.15,"cap":300},
    {"name":"Bridge_35","nim":0.0205,"rw":0.35,"cap":250},
    {"name":"Bridge_100","nim":0.051,"rw":1.0,"cap":250},
    {"name":"Bridge_150","nim":0.07,"rw":1.5,"cap":150},
    {"name":"Treasury","nim":0.0075,"rw":0.15,"cap":1000},
    {"name":"BTL","nim":0.010,"rw":0.35,"cap":200}
  ]
}"""

ASSET_ORDER = ["Lombard", "Bridge_35", "Bridge_100", "Bridge_150", "BTL", "Treasury"]


def is_binding(used: float, cap: float, rel: float = 0.005) -> bool:
    if cap is None or cap == 0:
        return False
    return (cap - used) <= abs(cap) * rel


def _validate_cfg(cfg: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        assert "capital" in cfg and "assets" in cfg, "Config must contain 'capital' and 'assets'."
        cap = cfg["capital"]
        for k in ["CET1", "leverage_ratio", "RWA_threshold"]:
            assert k in cap, f"Missing capital['{k}']."
            assert isinstance(cap[k], (int, float)), f"capital['{k}'] must be numeric."
        assets = cfg["assets"]
        assert isinstance(assets, list) and len(assets) > 0, "'assets' must be a non-empty list."
        for a in assets:
            for k in ["name", "nim", "rw", "cap"]:
                assert k in a, f"Each asset must include '{k}'."
            assert isinstance(a["name"], str), "Asset 'name' must be a string."
            for k in ["nim", "rw", "cap"]:
                assert isinstance(a[k], (int, float)), f"Asset '{a['name']}' field '{k}' must be numeric."
            assert a["nim"] >= 0 and a["rw"] >= 0 and a["cap"] >= 0
        return True, None
    except AssertionError as e:
        return False, str(e)


def _currency0(v: float) -> str:
    return f"£{v:,.0f}m" if pd.notna(v) else "—"


def _currency1(v: float) -> str:
    return f"£{v:,.1f}m" if pd.notna(v) else "—"


def _pct1(v: float) -> str:
    return f"{(100.0*v):.1f}%" if pd.notna(v) else "—"


# --------------------------- Optimisation Core ------------------------------ #

def build_and_solve(
    cfg: Dict[str, Any],
    *,
    CET1_override: Optional[float] = None,
    LEV_CAP_override: Optional[float] = None,
    RWA_threshold_override: Optional[float] = None,
    lam: float = 0.0,
) -> Tuple[str, Optional[float], Optional[pd.DataFrame], Dict[str, float], List[str]]:

    capital = cfg["capital"]
    CET1 = float(capital["CET1"]) if CET1_override is None else float(CET1_override)
    leverage_ratio = float(capital["leverage_ratio"])
    RWA_threshold = float(capital["RWA_threshold"]) if RWA_threshold_override is None else float(RWA_threshold_override)

    base_LEV_CAP = CET1 / leverage_ratio
    LEV_CAP = base_LEV_CAP if LEV_CAP_override is None else float(LEV_CAP_override)

    # Enforce asset order
    assets = sorted(cfg["assets"], key=lambda a: ASSET_ORDER.index(a["name"]) if a["name"] in ASSET_ORDER else 999)

    model = pulp.LpProblem(name="capital_allocation_optimiser", sense=pulp.LpMaximize)

    x_vars = {a["name"]: pulp.LpVariable(f"x_{a['name']}", lowBound=0) for a in assets}
    z = pulp.LpVariable("z", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)

    total_nii = pulp.lpSum(a["nim"] * x_vars[a["name"]] for a in assets)
    cet_used_expr = RATE_LOW * z + RATE_HIGH * y
    if lam and lam > 0:
        model += total_nii - lam * cet_used_expr
    else:
        model += total_nii

    for a in assets:
        model += x_vars[a["name"]] <= a["cap"], f"cap_{a['name']}"

    sum_x = pulp.lpSum(x_vars.values())
    model += sum_x <= LEV_CAP, "leverage_cap"

    rwa_expr = pulp.lpSum(a["rw"] * x_vars[a["name"]] for a in assets)
    model += z + y == rwa_expr, "rwa_split"
    model += z <= RWA_threshold, "z_threshold"
    model += cet_used_expr <= CET1, "cet_limit"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model_status = model.solve(solver)
    status_str = pulp.LpStatus.get(model_status, "Unknown")

    if status_str not in ("Optimal", "Feasible"):
        totals = {"LEV_CAP": LEV_CAP, "CET1": CET1, "RWA_threshold": RWA_threshold}
        return status_str, None, None, totals, []

    x_vals = {name: float(var.value() or 0.0) for name, var in x_vars.items()}
    z_val = float(z.value() or 0.0)
    y_val = float(y.value() or 0.0)
    total_exposure = sum(x_vals.values())
    rwa_total = sum(a["rw"] * x_vals[a["name"]] for a in assets)
    cet_used = RATE_LOW * z_val + RATE_HIGH * y_val
    obj_val = float(pulp.value(model.objective))

    bindings = []
    if is_binding(total_exposure, LEV_CAP):
        bindings.append("Leverage")
    if is_binding(cet_used, CET1):
        bindings.append("CET1")
    if is_binding(z_val, RWA_threshold) and y_val > 1e-9:
        bindings.append("RWA threshold")

    rows = []
    if rwa_total > 0:
        z_share = z_val / rwa_total
        y_share = y_val / rwa_total
    else:
        z_share = 0.0
        y_share = 0.0

    for a in assets:
        name = a["name"]
        exp = x_vals[name]
        nim = a["nim"]
        rw = a["rw"]
        nii = nim * exp
        rwa_i = rw * exp
        cet_alloc = RATE_LOW * (rwa_i * z_share) + RATE_HIGH * (rwa_i * y_share)
        roe = (nii / cet_alloc) if cet_alloc > 0 else float("nan")
        rows.append(
            {
                "Asset": name,
                "Exposure": exp,
                "NIM": nim,
                "RW": rw,
                "NII": nii,
                "Apportioned_CET_used": cet_alloc,
                "ROE": roe,
            }
        )

    df = pd.DataFrame(rows)

    # Totals row
    total_row = {
        "Asset": "TOTAL",
        "Exposure": df["Exposure"].sum(),
        "NIM": float("nan"),
        "RW": float("nan"),
        "NII": df["NII"].sum(),
        "Apportioned_CET_used": df["Apportioned_CET_used"].sum(),
        "ROE": (df["NII"].sum() / df["Apportioned_CET_used"].sum()) if df["Apportioned_CET_used"].sum() > 0 else float("nan"),
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    df["Asset"] = pd.Categorical(df["Asset"], categories=ASSET_ORDER + ["TOTAL"], ordered=True)
    df = df.sort_values("Asset").reset_index(drop=True)

    totals = {
        "total_exposure": total_exposure,
        "total_nii": sum(r["NII"] for r in rows),
        "LEV_CAP": LEV_CAP,
        "leverage_used_pct": (total_exposure / LEV_CAP) if LEV_CAP > 0 else float("nan"),
        "CET_used": cet_used,
        "CET1": CET1,
        "CET_used_pct": (cet_used / CET1) if CET1 > 0 else float("nan"),
        "RWA": rwa_total,
        "z": z_val,
        "y": y_val,
        "RWA_threshold": RWA_threshold,
        "objective": obj_val,
    }

    return status_str, obj_val, df, totals, bindings


def estimate_shadow_price(cfg: Dict[str, Any], which: str, delta: float, lam: float) -> Optional[float]:
    base_status, base_obj, _, _, _ = build_and_solve(cfg, lam=lam)
    if base_status not in ("Optimal", "Feasible") or base_obj is None:
        return None
    if which == "CET1":
        CET1_bumped = cfg["capital"]["CET1"] + delta
        stt, obj_b, *_ = build_and_solve(cfg, CET1_override=CET1_bumped, lam=lam)
    elif which == "LEV_CAP":
        CET1_same = cfg["capital"]["CET1"]
        base_LEV_CAP = CET1_same / cfg["capital"]["leverage_ratio"]
        stt, obj_b, *_ = build_and_solve(cfg, LEV_CAP_override=base_LEV_CAP + delta, lam=lam)
    elif which == "RWA_THRESHOLD":
        thr_bumped = cfg["capital"]["RWA_threshold"] + delta
        stt, obj_b, *_ = build_and_solve(cfg, RWA_threshold_override=thr_bumped, lam=lam)
    else:
        return None
    if stt not in ("Optimal", "Feasible") or obj_b is None:
        return None
    return (obj_b - base_obj) / delta


# ------------------------------- UI Layer ----------------------------------- #

st.set_page_config(page_title="Capital Allocation Optimiser", layout="wide")

st.title("Capital Allocation Optimiser")
st.caption(
    "Maximise portfolio NII with per-asset caps, a leverage cap, and a piecewise CET1 rule "
    f"({RATE_LOW*100:.1f}% up to the RWA threshold, {RATE_HIGH*100:.1f}% above)."
)

# --- Configuration source (JSON) ---
with st.expander("Advanced: Configuration JSON (editable)"):
    json_text = st.text_area(
        "Configuration JSON",
        value=DEFAULT_JSON,
        height=260,
        help="You can still edit raw JSON here. The quick inputs below override these values at solve time.",
    )
raw_json_for_download = json_text

# Parse JSON
try:
    cfg = json.loads(json_text)
    ok, err = _validate_cfg(cfg)
    if not ok:
        st.error(f"Validation error: {err}")
        st.stop()
except Exception as e:
    st.error(f"JSON parse error: {e}")
    st.stop()

# --- QUICK INPUTS ---
st.subheader("Quick inputs")

ci1, ci2, ci3, ci4 = st.columns([1, 1, 1, 1])
with ci1:
    q_CET1 = st.number_input("CET1 available", min_value=0.0, value=float(cfg["capital"]["CET1"]), step=1.0, help="Total CET1 you are willing to deploy (in £m).")
with ci2:
    q_leverage_ratio = st.number_input("Leverage ratio (CET1 / Exposure)", min_value=0.0001, value=float(cfg["capital"]["leverage_ratio"]), step=0.0005, format="%.4f", help="Regulatory leverage ratio. Exposure cap = CET1 / leverage ratio.")
with ci3:
    q_RWA_threshold = st.number_input("RWA threshold (split point)", min_value=0.0, value=float(cfg["capital"]["RWA_threshold"]), step=1.0, help=f"RWA up to this level uses {RATE_LOW*100:.1f}% CET1; above it uses {RATE_HIGH*100:.1f}%.")
with ci4:
    lam = st.slider("λ (capital penalty)", min_value=0.00, max_value=0.10, step=0.01, value=0.00,
                    help=("If > 0, the objective becomes NII − λ·CET_used.\n"
                          "Interpretation: λ is the 'price of CET1' in NII units.\n"
                          "Example: λ = 0.05 means the optimiser treats using £1m of CET1 "
                          "as costing £0.05m of NII in the objective, so it favours "
                          "allocations that save capital."))

st.markdown("**Per-asset caps**")
grid = st.columns(3)
cap_inputs = {}
assets_sorted = sorted(cfg["assets"], key=lambda a: ASSET_ORDER.index(a["name"]) if a["name"] in ASSET_ORDER else 999)
for i, a in enumerate(assets_sorted):
    with grid[i % 3]:
        cap_inputs[a["name"]] = st.number_input(
            f"{a['name']} cap (Exposure, £m)",
            min_value=0.0,
            value=float(a["cap"]),
            step=10.0,
            key=f"cap_{a['name']}",
            help=f"NIM={_pct1(a['nim'])}, RW={_pct1(a['rw'])}",
        )

st.markdown("---")
b1, b2 = st.columns([1, 3])
with b1:
    solve_clicked = st.button("Solve optimisation", type="primary")
with b2:
    st.download_button("Download current JSON (raw above)", data=raw_json_for_download.encode("utf-8"),
                       file_name="capital_optimiser_config.json", mime="application/json")

# Build effective cfg applying quick-input overrides
effective_cfg = {
    "capital": {
        "CET1": q_CET1,
        "leverage_ratio": q_leverage_ratio,
        "RWA_threshold": q_RWA_threshold,
    },
    "assets": [],
}
for a in assets_sorted:
    new_a = dict(a)
    new_a["cap"] = float(cap_inputs[a["name"]])
    effective_cfg["assets"].append(new_a)

# Main solve path
if solve_clicked:
    status, obj, df, totals, bindings = build_and_solve(effective_cfg, lam=lam)

    st.subheader("Optimisation Result")
    st.write(f"Status: **{status}**")
    if status not in ("Optimal", "Feasible") or df is None:
        st.warning("The model is infeasible or unbounded for the current inputs. Please review constraints and caps.")
        if totals:
            with st.expander("Diagnostics"):
                st.json(totals)
        st.stop()

    # KPI metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Exposure", _currency0(totals['total_exposure']))
    k2.metric("Total NII", _currency1(totals['total_nii']))
    k3.metric("Leverage used %", _pct1(totals['leverage_used_pct']),
              help=f"Exposure / LEV_CAP = {_currency0(totals['total_exposure'])} / {_currency0(totals['LEV_CAP'])}")
    k4.metric("CET used %", _pct1(totals['CET_used_pct']),
              help=f"CET_used / CET1 = {_currency1(totals['CET_used'])} / {_currency0(totals['CET1'])}")

    # Binding constraints badges
    st.subheader("Binding Constraints")
    if len(bindings) == 0:
        st.info("No constraints appear to be binding within 0.5% tolerance.")
    else:
        bcols = st.columns(len(bindings))
        for i, b in enumerate(bindings):
            bcols[i].success(b)

    with st.expander("Constraint details"):
        st.markdown(f"- **Leverage cap (Σx ≤ LEV_CAP):** Σx = **{_currency0(totals['total_exposure'])}**, LEV_CAP = **{_currency0(totals['LEV_CAP'])}**")
        st.markdown(f"- **CET1 ({RATE_LOW:.3f}·z + {RATE_HIGH:.3f}·y ≤ CET1):** CET_used = **{_currency1(totals['CET_used'])}**, CET1 = **{_currency0(totals['CET1'])}**")
        st.markdown(f"- **RWA split (z + y = RWA, z ≤ threshold):** z = **{_currency0(totals['z'])}**, y = **{_currency0(totals['y'])}**, RWA = **{_currency0(totals['RWA'])}**, threshold = **{_currency0(totals['RWA_threshold'])}**")

    # Table formatting
    display_df = df.copy()
    display_df["Exposure"] = display_df["Exposure"].map(_currency0)
    display_df["NIM"] = display_df["NIM"].map(_pct1)
    display_df["RW"] = display_df["RW"].map(_pct1)
    display_df["NII"] = display_df["NII"].map(_currency1)
    display_df["Apportioned_CET_used"] = display_df["Apportioned_CET_used"].map(_currency1)
    display_df["ROE"] = display_df["ROE"].map(_pct1)

    def highlight_total(row):
        return ['font-weight: bold' if row["Asset"] == "TOTAL" else '' for _ in row]

    st.subheader("Allocation by Asset")
    st.dataframe(display_df.style.apply(highlight_total, axis=1), use_container_width=True, hide_index=True)

    # Download allocation CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download allocation CSV", data=csv_bytes, file_name="capital_allocation.csv", mime="text/csv")

    # Chart
    chart_df = df[df["Asset"] != "TOTAL"][["Asset", "Exposure"]].set_index("Asset")
    st.subheader("Exposure by Asset")
    st.bar_chart(chart_df)

    # CET1 marginal value
    st.subheader("CET1 marginal value (approx.)")
    st.caption("ℹ️ **What is this?** We bump CET1 by **+£1m**, re‑solve the optimisation, and show the **extra NII** the model would deliver. "
               "If the number is near zero, CET1 is not binding. If positive, it is the marginal 'price' of CET1 in £m NII per +£1m CET1.")
    sp_cet = estimate_shadow_price(effective_cfg, which="CET1", delta=1.0, lam=lam)  # per +£1m CET1
    st.metric("Incremental NII for +£1m CET1", _currency1(sp_cet) if pd.notna(sp_cet) else "n/a")

else:
    st.info("Set CET1 and per-asset caps above, then click **Solve optimisation**. Use the JSON expander for advanced edits (NIM/RW, add/remove assets).")

st.markdown("---")
st.caption("Binding means utilisation within 0.5% of the cap. Monetary figures shown in £m. "
           "Piecewise CET1 is 14.8% up to the threshold and 17.3% above.")
