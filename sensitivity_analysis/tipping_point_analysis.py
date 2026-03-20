"""
Tipping-Point Sensitivity Analysis
====================================
Core question: "How much worse would missing outcomes in the EPS arm
need to be (relative to MAR imputation) before the EPS vs Human
treatment effect loses statistical significance?"

Primary scenario: EPS-arm-specific delta (most conservative for EPS benefit).
Secondary: differential delta (EPS vs Human arms).

Reports both: p > 0.05 crossing AND CI crossing zero.

Requirements:
  pip install pandas numpy scipy openpyxl statsmodels scikit-learn
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from scipy.stats import t as tdist
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer


# ============================================================
# Helpers
# ============================================================
def clean_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def clean_sex(series):
    raw = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    out[raw.isin({"女", "female", "f", "0", "2"})] = 1.0
    out[raw.isin({"男", "male", "m", "1"})]         = 0.0
    return out


def pick_col(df, candidates):
    mapping = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    for c in candidates:
        if c.lower() in mapping:
            return mapping[c.lower()]
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt_p(p):
    return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")


def ancova_effect(dc, outcome, covars):
    covars = [c for c in covars if c in dc.columns]
    cols = ["group_num"] + covars + [outcome]
    d = dc[cols].dropna()
    if len(d) < len(cols) + 3:
        return np.nan, np.nan, np.nan
    y = d[outcome].astype(float)
    X = sm.add_constant(d[["group_num"] + covars].astype(float))
    try:
        m = sm.OLS(y, X).fit()
        return float(m.params["group_num"]), float(m.bse["group_num"]), float(m.pvalues["group_num"])
    except Exception:
        return np.nan, np.nan, np.nan


def rubins_rules(estimates, variances):
    est = np.array(estimates, dtype=float)
    var = np.array(variances, dtype=float)
    ok = ~(np.isnan(est) | np.isnan(var))
    est, var = est[ok], var[ok]
    m = len(est)
    if m < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    Q = np.mean(est); U = np.mean(var); B = np.var(est, ddof=1)
    T = U + (1 + 1/m) * B
    se = math.sqrt(T) if T > 0 else np.nan
    if B > 0 and U > 0:
        r = (1 + 1/m) * B / U
        df = (m - 1) * (1 + 1/r)**2
    else:
        df = 1e6
    tcrit = float(tdist.ppf(0.975, df))
    lo = Q - tcrit * se if not np.isnan(se) else np.nan
    hi = Q + tcrit * se if not np.isnan(se) else np.nan
    p = 2 * float(tdist.sf(abs(Q / se), df)) if se and se > 0 else np.nan
    return Q, lo, hi, p, se


# ============================================================
# Data loading
# ============================================================
def load_wl_arm(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    col_age  = pick_col(df, ["age", "年龄"])
    col_sex  = pick_col(df, ["sex", "性别"])
    col_bmi  = pick_col(df, ["BMI", "bmi"])
    col_bw   = pick_col(df, ["baseline_weight_kg", "入营体重", "入营体重kg"])
    col_ht   = pick_col(df, ["height", "身高"])
    col_hba1c = pick_col(df, ["hba1c", "HbA1c", "糖化血红蛋白"])
    col_wl   = pick_col(df, ["weight_loss_kg", "减重数"])
    d = pd.DataFrame({
        "age":   clean_numeric(df[col_age]).values if col_age else np.nan,
        "sex":   clean_sex(df[col_sex]).values if col_sex else np.nan,
        "bmi":   clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
        "height": clean_numeric(df[col_ht]).values if col_ht else np.nan,
        "bw":    clean_numeric(df[col_bw]).values if col_bw else np.nan,
        "hba1c": clean_numeric(df[col_hba1c]).values if col_hba1c else np.nan,
        "wl_kg": clean_numeric(df[col_wl]).values if col_wl else np.nan,
    })
    d["group"] = label; d["completer"] = 1
    d["wl_pct"] = np.where(d["bw"] > 0, d["wl_kg"] / d["bw"] * 100, np.nan)
    return d


def load_gl_arm(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    col_age  = pick_col(df, ["age", "年龄"])
    col_sex  = pick_col(df, ["sex", "性别"])
    col_bmi  = pick_col(df, ["BMI", "bmi"])
    col_fpg0 = pick_col(df, ["baseline_fpg_mmol", "入营空腹"])
    col_fpg1 = pick_col(df, ["endpoint_fpg_mmol", "结营空腹"])
    col_ppg0 = pick_col(df, ["baseline_ppg_mmol", "入营餐后2小时"])
    d = pd.DataFrame({
        "age":  clean_numeric(df[col_age]).values if col_age else np.nan,
        "sex":  clean_sex(df[col_sex]).values if col_sex else np.nan,
        "bmi":  clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
        "fpg0": clean_numeric(df[col_fpg0]).values if col_fpg0 else np.nan,
        "ppg0": clean_numeric(df[col_ppg0]).values if col_ppg0 else np.nan,
        "fpg1": clean_numeric(df[col_fpg1]).values if col_fpg1 else np.nan,
    })
    d["group"] = label; d["completer"] = 1
    d["fpg_change"] = d["fpg0"] - d["fpg1"]
    return d


# ============================================================
# ITT population helpers
# ============================================================
def gen_miss(comp, n_miss, rng, bl_cols):
    if n_miss <= 0:
        return pd.DataFrame(columns=comp.columns)
    available_bl = [c for c in bl_cols if c in comp.columns]
    s = comp[available_bl].sample(n=n_miss, replace=True, random_state=rng).reset_index(drop=True)
    for c in [x for x in available_bl if x != "group"]:
        v = s[c].notna()
        if v.any():
            sd = comp[c].std() * 0.1
            if pd.notna(sd) and sd > 0:
                s[c] = s[c].astype(float)
                s.loc[v, c] += rng.normal(0, sd, size=v.sum())
    return s


def gen_miss_wl(comp, n_miss, rng):
    bl = ["group", "age", "sex", "bmi", "height", "bw", "hba1c"]
    s = gen_miss(comp, n_miss, rng, bl)
    if len(s) > 0:
        s["wl_kg"] = np.nan; s["wl_pct"] = np.nan; s["completer"] = 0
    return s


def gen_miss_gl(comp, n_miss, rng):
    bl = ["group", "age", "sex", "bmi", "fpg0", "ppg0"]
    s = gen_miss(comp, n_miss, rng, bl)
    if len(s) > 0:
        s["fpg1"] = np.nan; s["fpg_change"] = np.nan; s["completer"] = 0
    return s


# ============================================================
# MICE functions
# ============================================================
def mice_wl(df_full, m=20, seed=42):
    candidate = ["group_num", "age", "sex", "bmi", "height", "bw", "hba1c", "wl_kg"]
    cols = [c for c in candidate if c in df_full.columns and df_full[c].notna().any()]
    base = df_full[cols].copy()
    out = []
    for i in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=50,
                               random_state=seed + i, sample_posterior=True, skip_complete=True)
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=cols)
        dc = df_full.copy()
        dc["wl_kg"] = filled["wl_kg"]
        bw = filled["bw"] if "bw" in filled.columns else dc["bw"]
        dc["wl_pct"] = np.where(bw > 0, dc["wl_kg"] / bw * 100, np.nan)
        out.append(dc)
    return out


def mice_gl(df_full, m=20, seed=42):
    candidate = ["group_num", "age", "sex", "bmi", "fpg0", "ppg0", "fpg1"]
    cols = [c for c in candidate if c in df_full.columns and df_full[c].notna().any()]
    base = df_full[cols].copy()
    out = []
    for i in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=50,
                               random_state=seed + i, sample_posterior=True, skip_complete=True)
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=cols)
        dc = df_full.copy()
        dc["fpg1"] = filled["fpg1"]
        dc["fpg_change"] = dc["fpg0"] - dc["fpg1"]
        out.append(dc)
    return out


# ============================================================
# Tipping-point search
# ============================================================
def tipping_search(df_itt, mice_func, outcome_col, delta_range, covars,
                   apply_to="eps_only", delta_target="wl_kg", m=20, seed=42):
    base_datasets = mice_func(df_itt, m=m, seed=seed)
    results = []

    for delta in delta_range:
        coefs, vars_ = [], []
        for dc in base_datasets:
            ds = dc.copy()
            miss = df_itt["completer"] == 0
            if apply_to == "eps_only":
                mask = miss & (df_itt["group"] == "EPS")
            elif apply_to == "human_only":
                mask = miss & (df_itt["group"] == "Human")
            else:
                mask = miss

            if delta_target == "wl_kg":
                ds.loc[mask, "wl_kg"] += delta
                bw = ds["bw"]
                ds["wl_pct"] = np.where(bw > 0, ds["wl_kg"] / bw * 100, np.nan)
            elif delta_target == "fpg1":
                ds.loc[mask, "fpg1"] += delta
                ds["fpg_change"] = ds["fpg0"] - ds["fpg1"]

            c, se, p = ancova_effect(ds, outcome_col, covars)
            coefs.append(c)
            vars_.append(se**2 if not np.isnan(se) else np.nan)

        Q, lo, hi, p, se = rubins_rules(coefs, vars_)
        sig = p < 0.05 if not np.isnan(p) else None
        ci_excl = (lo > 0 or hi < 0) if not (np.isnan(lo) or np.isnan(hi)) else None

        results.append({
            "delta": delta, "estimate": Q, "ci_lo": lo, "ci_hi": hi,
            "p": p, "significant": sig, "ci_excludes_zero": ci_excl,
        })
    return results


def find_tipping(results):
    for r in results:
        if r["significant"] is False or r["ci_excludes_zero"] is False:
            return r["delta"], r["p"], r["ci_lo"], r["ci_hi"]
    return None, None, None, None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tipping-point sensitivity analysis for weight-loss and glycemic cohorts")
    parser.add_argument("--weight_human", type=str,
                        default="data/example/weight_loss/human_arm.xlsx")
    parser.add_argument("--weight_eps", type=str,
                        default="data/example/weight_loss/eps_arm.xlsx")
    parser.add_argument("--gly_human", type=str,
                        default="data/example/glycemic/human_arm.xlsx")
    parser.add_argument("--gly_eps", type=str,
                        default="data/example/glycemic/eps_arm.xlsx")
    parser.add_argument("--n_rand_human_wl", type=int, default=100)
    parser.add_argument("--n_rand_eps_wl", type=int, default=100)
    parser.add_argument("--n_rand_human_gl", type=int, default=50)
    parser.add_argument("--n_rand_eps_gl", type=int, default=50)
    parser.add_argument("--m_imputations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default="outputs/sensitivity_analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_xlsx = out_dir / "tipping_point_results.xlsx"

    M = args.m_imputations
    SEED = args.seed

    # ========== Weight Loss ==========
    print("=" * 60)
    print("TIPPING-POINT SENSITIVITY ANALYSIS")
    print("=" * 60)

    comp_e_wl = load_wl_arm(args.weight_eps,   "EPS")
    comp_h_wl = load_wl_arm(args.weight_human, "Human")

    rng = np.random.default_rng(SEED)
    n_miss_h_wl = max(0, args.n_rand_human_wl - len(comp_h_wl))
    n_miss_e_wl = max(0, args.n_rand_eps_wl   - len(comp_e_wl))

    df_itt_wl = pd.concat([comp_h_wl, comp_e_wl,
                           gen_miss_wl(comp_h_wl, n_miss_h_wl, rng),
                           gen_miss_wl(comp_e_wl, n_miss_e_wl, rng)], ignore_index=True)
    df_itt_wl["group_num"] = (df_itt_wl["group"] == "EPS").astype(float)

    # ========== Glycemic ==========
    comp_h_gl = load_gl_arm(args.gly_human, "Human")
    comp_e_gl = load_gl_arm(args.gly_eps,   "EPS")

    n_miss_h_gl = max(0, args.n_rand_human_gl - len(comp_h_gl))
    n_miss_e_gl = max(0, args.n_rand_eps_gl   - len(comp_e_gl))

    df_itt_gl = pd.concat([comp_h_gl, comp_e_gl,
                           gen_miss_gl(comp_h_gl, n_miss_h_gl, rng),
                           gen_miss_gl(comp_e_gl, n_miss_e_gl, rng)], ignore_index=True)
    df_itt_gl["group_num"] = (df_itt_gl["group"] == "EPS").astype(float)

    # ========== Tipping-point searches ==========
    wl_covars = ["bw", "age", "sex", "bmi"]
    gl_covars = ["fpg0", "age", "sex", "bmi"]

    print("\n--- Weight Loss Tipping Points ---")
    wl_delta_range = np.arange(0, -10.1, -0.5)

    print("  A) EPS dropouts only → wl_kg outcome...")
    tp_wl_eps_kg = tipping_search(df_itt_wl, mice_wl, "wl_kg", wl_delta_range, wl_covars,
                                   "eps_only", "wl_kg", m=M, seed=SEED)

    print("  B) EPS dropouts only → wl_pct outcome...")
    tp_wl_eps_pct = tipping_search(df_itt_wl, mice_wl, "wl_pct", wl_delta_range, wl_covars,
                                    "eps_only", "wl_kg", m=M, seed=SEED)

    print("  C) Differential: EPS worse, Human better (±delta)...")
    tp_wl_diff = []
    base_wl = mice_wl(df_itt_wl, m=M, seed=SEED)
    for delta in wl_delta_range:
        coefs, vars_ = [], []
        for dc in base_wl:
            ds = dc.copy()
            miss = df_itt_wl["completer"] == 0
            eps_miss = miss & (df_itt_wl["group"] == "EPS")
            hum_miss = miss & (df_itt_wl["group"] == "Human")
            ds.loc[eps_miss, "wl_kg"] += delta
            ds.loc[hum_miss, "wl_kg"] -= delta
            bw = ds["bw"]
            ds["wl_pct"] = np.where(bw > 0, ds["wl_kg"] / bw * 100, np.nan)
            c, se, p = ancova_effect(ds, "wl_kg", wl_covars)
            coefs.append(c)
            vars_.append(se**2 if not np.isnan(se) else np.nan)
        Q, lo, hi, p, se = rubins_rules(coefs, vars_)
        tp_wl_diff.append({
            "delta": delta, "estimate": Q, "ci_lo": lo, "ci_hi": hi, "p": p,
            "significant": p < 0.05 if not np.isnan(p) else None,
            "ci_excludes_zero": (lo > 0 or hi < 0) if not (np.isnan(lo) or np.isnan(hi)) else None,
        })

    print("\n--- Glycemic Tipping Points ---")
    gl_delta_range = np.arange(0, 5.1, 0.2)

    print("  D) EPS dropouts only → fpg_change outcome...")
    tp_gl_eps = tipping_search(df_itt_gl, mice_gl, "fpg_change", gl_delta_range, gl_covars,
                                "eps_only", "fpg1", m=M, seed=SEED)

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("TIPPING POINT SUMMARY")
    print("=" * 60)

    analyses = [
        ("WL (kg) — EPS dropouts worse",       tp_wl_eps_kg,  "kg"),
        ("WL (%) — EPS dropouts worse",        tp_wl_eps_pct, "kg (applied to wl_kg)"),
        ("WL (kg) — Differential (EPS worse, Human better)", tp_wl_diff, "kg each direction"),
        ("FPG change — EPS dropouts worse",     tp_gl_eps,     "mmol/L on endline FPG"),
    ]

    summary_rows = []
    for name, res, unit in analyses:
        d, p, lo, hi = find_tipping(res)
        if d is not None:
            print(f"  {name}: tipping at delta={d:.1f} {unit}, p={p:.4f}, CI=({lo:.3f}, {hi:.3f})")
            summary_rows.append({
                "Analysis": name, "Tipping delta": f"{d:.1f}", "Unit": unit,
                "P at tipping": f"{p:.4f}", "CI at tipping": f"({lo:.3f}, {hi:.3f})",
                "Interpretation": f"Result non-significant when EPS dropouts are {abs(d):.1f} {unit} worse than MAR",
            })
        else:
            print(f"  {name}: robust across all deltas tested (up to {res[-1]['delta']:.1f})")
            summary_rows.append({
                "Analysis": name, "Tipping delta": f"Beyond {res[-1]['delta']:.1f}", "Unit": unit,
                "P at tipping": "N/A", "CI at tipping": "N/A",
                "Interpretation": "Robust: significant across all tested deltas",
            })

    # ========== Save ==========
    df_summary = pd.DataFrame(summary_rows)

    def detail_df(results):
        rows = []
        for r in results:
            rows.append({
                "Delta": r["delta"],
                "ANCOVA Estimate (EPS-Human)": f"{r['estimate']:.4f}" if not np.isnan(r["estimate"]) else "",
                "CI low": f"{r['ci_lo']:.4f}" if not np.isnan(r["ci_lo"]) else "",
                "CI high": f"{r['ci_hi']:.4f}" if not np.isnan(r["ci_hi"]) else "",
                "P value": fmt_p(r["p"]),
                "Significant (p<0.05)": r["significant"],
                "CI excludes 0": r["ci_excludes_zero"],
            })
        return pd.DataFrame(rows)

    df_wl_kg  = detail_df(tp_wl_eps_kg)
    df_wl_pct = detail_df(tp_wl_eps_pct)
    df_wl_dif = detail_df(tp_wl_diff)
    df_gl     = detail_df(tp_gl_eps)

    notes = pd.DataFrame([
        {"Item": "Purpose",             "Details": "How far from MAR must missing data be to nullify the EPS benefit?"},
        {"Item": "Primary scenario",    "Details": "Delta applied to EPS-arm dropouts only (most conservative)"},
        {"Item": "Differential scenario", "Details": "EPS dropouts worse by delta, Human dropouts better by delta (extreme)"},
        {"Item": "WL delta unit",       "Details": "kg, applied to wl_kg; wl_pct re-derived from wl_kg/baseline_wt"},
        {"Item": "Glycemic delta unit", "Details": "mmol/L, applied to endline FPG; fpg_change re-derived"},
        {"Item": "Tipping criterion",   "Details": "p > 0.05 OR 95% CI includes zero"},
        {"Item": "Analysis model",      "Details": "ANCOVA pooled via Rubin's rules"},
        {"Item": "Imputations per delta", "Details": str(M)},
    ])

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_wl_kg.to_excel(writer,  sheet_name="WL kg EPS-only", index=False)
        df_wl_pct.to_excel(writer, sheet_name="WL pct EPS-only", index=False)
        df_wl_dif.to_excel(writer, sheet_name="WL kg Differential", index=False)
        df_gl.to_excel(writer,     sheet_name="FPG EPS-only", index=False)
        notes.to_excel(writer,     sheet_name="Notes", index=False)

    print(f"\n✅ Saved to: {out_xlsx}")


if __name__ == "__main__":
    main()
