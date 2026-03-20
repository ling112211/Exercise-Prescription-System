"""
ITT Sensitivity Analysis — Glycemic Control Cohort (Exploratory)
=================================================================
Positioning:  EXPLORATORY supplementary sensitivity analysis.
              Glycemic cohort is small; results should be presented in
              supplementary material only.

Workflow:
  1. Construct ITT population from completers + bootstrapped missing
  2. MICE imputation of endline FPG ONLY → derive change scores
  3. ANCOVA on each imputed dataset → pool via Rubin's rules
  4. MNAR delta on endline FPG (EPS dropouts, mmol/L units)
  5. BOCF (missing endline FPG = baseline FPG, i.e. 0 change)
  6. MI diagnostics

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


def fmt(v, d=3):
    return "" if pd.isna(v) else f"{v:.{d}f}"

def fmt_ci(e, lo, hi, d=3):
    return "" if any(pd.isna(x) for x in [e, lo, hi]) else f"{e:.{d}f} ({lo:.{d}f}, {hi:.{d}f})"

def fmt_p(p):
    return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")


# ============================================================
# Load data
# ============================================================
def load_glycemic_arm(path, label):
    """Load one arm's glycemic data from Excel with flexible column names."""
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age  = pick_col(df, ["age", "年龄"])
    col_sex  = pick_col(df, ["sex", "性别"])
    col_bmi  = pick_col(df, ["BMI", "bmi"])
    col_fpg0 = pick_col(df, ["baseline_fpg_mmol", "入营空腹"])
    col_fpg1 = pick_col(df, ["endpoint_fpg_mmol", "结营空腹"])
    col_ppg0 = pick_col(df, ["baseline_ppg_mmol", "入营餐后2小时"])
    col_bw   = pick_col(df, ["baseline_weight_kg", "入营体重", "入营体重kg", "初始体重（档案）"])

    d = pd.DataFrame({
        "age":  clean_numeric(df[col_age]).values if col_age else np.nan,
        "sex":  clean_sex(df[col_sex]).values if col_sex else np.nan,
        "bmi":  clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
        "bw":   clean_numeric(df[col_bw]).values if col_bw else np.nan,
        "fpg0": clean_numeric(df[col_fpg0]).values if col_fpg0 else np.nan,
        "ppg0": clean_numeric(df[col_ppg0]).values if col_ppg0 else np.nan,
        "fpg1": clean_numeric(df[col_fpg1]).values if col_fpg1 else np.nan,
    })
    d["group"] = label
    d["completer"] = 1
    d["fpg_change"]     = d["fpg0"] - d["fpg1"]  # positive = improvement
    d["fpg_change_pct"] = np.where(d["fpg0"] > 0, d["fpg_change"] / d["fpg0"] * 100, np.nan)
    return d


# ============================================================
# ITT population construction
# ============================================================
BASELINE_COLS = ["group", "age", "sex", "bmi", "bw", "fpg0", "ppg0"]


def generate_missing(comp_df, n_miss, rng):
    if n_miss <= 0:
        return pd.DataFrame(columns=comp_df.columns)
    available_bl = [c for c in BASELINE_COLS if c in comp_df.columns]
    sampled = comp_df[available_bl].sample(n=n_miss, replace=True, random_state=rng).reset_index(drop=True)
    for col in ["age", "bmi", "bw", "fpg0", "ppg0"]:
        if col not in sampled.columns:
            continue
        v = sampled[col].notna()
        if v.any():
            sd = comp_df[col].std() * 0.1
            if pd.notna(sd) and sd > 0:
                sampled[col] = sampled[col].astype(float)
                sampled.loc[v, col] += rng.normal(0, sd, size=v.sum())
    sampled["fpg1"] = np.nan
    sampled["fpg_change"] = np.nan
    sampled["fpg_change_pct"] = np.nan
    sampled["completer"] = 0
    return sampled


# ============================================================
# MICE
# ============================================================
def run_mice(df_full, m=20, seed=42):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.linear_model import BayesianRidge
    from sklearn.impute import IterativeImputer

    candidate_covars = ["group_num", "age", "sex", "bmi", "fpg0", "ppg0"]
    mi_covars = [c for c in candidate_covars if c in df_full.columns and df_full[c].notna().any()]
    impute_cols = mi_covars + ["fpg1"]

    base = df_full[impute_cols].copy()
    datasets = []
    for i in range(m):
        imp = IterativeImputer(
            estimator=BayesianRidge(), max_iter=50,
            random_state=seed + i, sample_posterior=True, skip_complete=True,
        )
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=impute_cols)
        dc = df_full.copy()
        dc["fpg1"] = filled["fpg1"]
        dc["fpg_change"] = dc["fpg0"] - dc["fpg1"]
        dc["fpg_change_pct"] = np.where(dc["fpg0"] > 0,
                                         dc["fpg_change"] / dc["fpg0"] * 100, np.nan)
        datasets.append(dc)
    return datasets


# ============================================================
# Rubin's rules + ANCOVA
# ============================================================
def rubins_rules(estimates, variances):
    from scipy.stats import t as tdist
    est = np.array(estimates, dtype=float)
    var = np.array(variances, dtype=float)
    valid = ~(np.isnan(est) | np.isnan(var))
    est, var = est[valid], var[valid]
    m = len(est)
    if m < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    Q = np.mean(est); U = np.mean(var); B = np.var(est, ddof=1)
    T = U + (1 + 1/m) * B
    se = math.sqrt(T) if T > 0 else np.nan
    fmi = ((1 + 1/m) * B / T) if T > 0 else np.nan

    if B > 0 and U > 0:
        r = (1 + 1/m) * B / U
        df = (m - 1) * (1 + 1/r)**2
    else:
        df = 1e6

    tcrit = float(tdist.ppf(0.975, df))
    ci_lo = Q - tcrit * se if not np.isnan(se) else np.nan
    ci_hi = Q + tcrit * se if not np.isnan(se) else np.nan
    p = 2 * float(tdist.sf(abs(Q / se), df)) if se and se > 0 else np.nan
    return Q, se, ci_lo, ci_hi, p, df, fmi


def ancova_effect(dc, outcome, covars=None):
    import statsmodels.api as sm
    if covars is None:
        covars = ["fpg0", "age", "sex", "bmi"]
    covars = [c for c in covars if c in dc.columns]
    cols = ["group_num"] + covars + [outcome]
    d = dc[cols].dropna()
    if len(d) < len(cols) + 3:
        return np.nan, np.nan, np.nan, np.nan
    y = d[outcome].astype(float)
    X = sm.add_constant(d[["group_num"] + covars].astype(float))
    try:
        model = sm.OLS(y, X).fit()
        c = float(model.params["group_num"])
        se = float(model.bse["group_num"])
        p = float(model.pvalues["group_num"])
        return c, se, se**2, p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def pool_ancova(datasets, outcome, covars=None):
    coefs, variances = [], []
    eps_m, eps_v, hum_m, hum_v = [], [], [], []
    for dc in datasets:
        c, se, v, p = ancova_effect(dc, outcome, covars)
        coefs.append(c); variances.append(v)
        e = dc.loc[dc.group == "EPS", outcome].dropna()
        h = dc.loc[dc.group == "Human", outcome].dropna()
        eps_m.append(e.mean())
        eps_v.append((e.std(ddof=1) / math.sqrt(len(e)))**2 if len(e) > 1 else np.nan)
        hum_m.append(h.mean())
        hum_v.append((h.std(ddof=1) / math.sqrt(len(h)))**2 if len(h) > 1 else np.nan)

    Q, se, lo, hi, p, df, fmi = rubins_rules(coefs, variances)
    eQ, _, eLo, eHi, _, _, _ = rubins_rules(eps_m, eps_v)
    hQ, _, hLo, hHi, _, _, _ = rubins_rules(hum_m, hum_v)
    return {
        "diff": {"est": Q, "se": se, "lo": lo, "hi": hi, "p": p, "fmi": fmi},
        "eps_mean": {"est": eQ, "lo": eLo, "hi": eHi},
        "hum_mean": {"est": hQ, "lo": hLo, "hi": hHi},
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="ITT sensitivity analysis — glycemic control cohort (exploratory)")
    parser.add_argument("--gly_human", type=str,
                        default="data/example/glycemic/human_arm.xlsx",
                        help="Path to Human-arm glycemic Excel file")
    parser.add_argument("--gly_eps", type=str,
                        default="data/example/glycemic/eps_arm.xlsx",
                        help="Path to EPS-arm glycemic Excel file")
    parser.add_argument("--n_randomized_human", type=int, default=50,
                        help="Number randomized in Human arm (default 50 for example data)")
    parser.add_argument("--n_randomized_eps", type=int, default=50,
                        help="Number randomized in EPS arm (default 50 for example data)")
    parser.add_argument("--m_imputations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default="outputs/sensitivity_analysis",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_xlsx = out_dir / "ITT_glycemic_results.xlsx"

    M = args.m_imputations
    SEED = args.seed
    N_RAND_H = args.n_randomized_human
    N_RAND_E = args.n_randomized_eps

    # --- Load ---
    comp_h = load_glycemic_arm(args.gly_human, "Human")
    comp_e = load_glycemic_arm(args.gly_eps,   "EPS")
    print(f"Completers: Human={len(comp_h)}, EPS={len(comp_e)}")

    # --- ITT population ---
    n_miss_h = max(0, N_RAND_H - len(comp_h))
    n_miss_e = max(0, N_RAND_E - len(comp_e))
    print(f"Missing: Human={n_miss_h}, EPS={n_miss_e}")

    rng = np.random.default_rng(SEED)
    df_itt = pd.concat([comp_h, comp_e,
                        generate_missing(comp_h, n_miss_h, rng),
                        generate_missing(comp_e, n_miss_e, rng)], ignore_index=True)
    df_itt["group_num"] = (df_itt["group"] == "EPS").astype(float)

    print(f"ITT: N={len(df_itt)} (Human={sum(df_itt.group=='Human')}, EPS={sum(df_itt.group=='EPS')})")
    print(f"Missing fpg1: {df_itt['fpg1'].isna().sum()}")

    # --- MICE (MAR) ---
    print("\nRunning MICE (MAR)...")
    imputed_mar = run_mice(df_itt, m=M, seed=SEED)
    print(f"  {len(imputed_mar)} imputed datasets.")

    # --- ITT ANCOVA ---
    print("\n=== ITT — ANCOVA (MAR MI) ===")
    res_fpg     = pool_ancova(imputed_mar, "fpg_change")
    res_fpg_pct = pool_ancova(imputed_mar, "fpg_change_pct")

    for label, r in [("FPG change (mmol/L)", res_fpg), ("FPG change (%)", res_fpg_pct)]:
        d = r["diff"]
        print(f"  {label}: diff={fmt_ci(d['est'], d['lo'], d['hi'])}, p={fmt_p(d['p'])}, FMI={fmt(d['fmi'], 2)}")

    # --- Available-case ANCOVA ---
    print("\n=== Available-case — ANCOVA ===")
    from scipy.stats import t as tdist
    df_ac = pd.concat([comp_h, comp_e], ignore_index=True)
    df_ac["group_num"] = (df_ac["group"] == "EPS").astype(float)

    ac_results = {}
    for outcome, label in [("fpg_change", "FPG change"), ("fpg_change_pct", "FPG change %")]:
        c, se, v, p = ancova_effect(df_ac, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_ac) - 6, 1)))
        ac_results[outcome] = {
            "diff": c,
            "lo": c - tcrit * se if not np.isnan(se) else np.nan,
            "hi": c + tcrit * se if not np.isnan(se) else np.nan,
            "p": p,
            "n_eps": len(comp_e), "n_hum": len(comp_h),
            "eps_mean": comp_e[outcome].mean(), "hum_mean": comp_h[outcome].mean(),
        }
        print(f"  {label}: diff={fmt_ci(c, ac_results[outcome]['lo'], ac_results[outcome]['hi'])}, p={fmt_p(p)}")

    # --- MNAR delta ---
    print("\n=== MNAR Delta (EPS dropouts, shift on endline FPG) ===")
    delta_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0]
    delta_results = {}
    for delta in delta_values:
        shifted = []
        for dc in run_mice(df_itt, m=10, seed=SEED):
            ds = dc.copy()
            eps_miss = (df_itt["completer"] == 0) & (df_itt["group"] == "EPS")
            ds.loc[eps_miss, "fpg1"] += delta
            ds["fpg_change"] = ds["fpg0"] - ds["fpg1"]
            ds["fpg_change_pct"] = np.where(ds["fpg0"] > 0, ds["fpg_change"] / ds["fpg0"] * 100, np.nan)
            shifted.append(ds)
        r_fpg = pool_ancova(shifted, "fpg_change")
        delta_results[delta] = r_fpg
        d = r_fpg["diff"]
        lbl = "MAR" if delta == 0 else f"+{delta:.1f} mmol/L"
        print(f"  {lbl}: diff={fmt_ci(d['est'], d['lo'], d['hi'])}, p={fmt_p(d['p'])}")

    # --- BOCF ---
    print("\n=== BOCF (missing = 0 FPG change) ===")
    df_bocf = df_itt.copy()
    miss = df_bocf["completer"] == 0
    df_bocf.loc[miss, "fpg1"] = df_bocf.loc[miss, "fpg0"]
    df_bocf["fpg_change"] = df_bocf["fpg0"] - df_bocf["fpg1"]
    df_bocf["fpg_change_pct"] = np.where(df_bocf["fpg0"] > 0,
                                          df_bocf["fpg_change"] / df_bocf["fpg0"] * 100, np.nan)

    bocf_results = {}
    for outcome in ["fpg_change", "fpg_change_pct"]:
        c, se, v, p = ancova_effect(df_bocf, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_bocf) - 6, 1)))
        bocf_results[outcome] = {
            "diff": c,
            "lo": c - tcrit * se if not np.isnan(se) else np.nan,
            "hi": c + tcrit * se if not np.isnan(se) else np.nan,
            "p": p,
        }
        print(f"  {outcome}: diff={fmt_ci(c, bocf_results[outcome]['lo'], bocf_results[outcome]['hi'])}, p={fmt_p(p)}")

    # --- MI diagnostics ---
    obs_fpg1 = df_itt.loc[df_itt.completer == 1, "fpg1"].dropna()
    imp_fpg1s = []
    for dc in imputed_mar:
        imp_fpg1s.extend(dc.loc[df_itt.completer == 0, "fpg1"].dropna().values)
    imp_fpg1 = np.array(imp_fpg1s)

    diag_rows = [
        {"Metric": "Observed endline FPG — N",   "Value": len(obs_fpg1)},
        {"Metric": "Observed endline FPG — Mean", "Value": f"{obs_fpg1.mean():.3f}"},
        {"Metric": "Observed endline FPG — SD",   "Value": f"{obs_fpg1.std():.3f}"},
        {"Metric": "Imputed endline FPG — N",     "Value": f"{len(imp_fpg1)} (across {M} imps)"},
        {"Metric": "Imputed endline FPG — Mean",  "Value": f"{imp_fpg1.mean():.3f}" if len(imp_fpg1) > 0 else "N/A"},
        {"Metric": "Imputed endline FPG — SD",    "Value": f"{imp_fpg1.std():.3f}" if len(imp_fpg1) > 0 else "N/A"},
        {"Metric": "FMI (fpg_change ANCOVA)",     "Value": fmt(res_fpg["diff"]["fmi"], 2)},
        {"Metric": "FMI (fpg_change_pct ANCOVA)", "Value": fmt(res_fpg_pct["diff"]["fmi"], 2)},
    ]

    # --- Save ---
    rows_main = []
    for outcome, label, r_itt, ac, bocf in [
        ("fpg_change", "FPG reduction (mmol/L)", res_fpg, ac_results["fpg_change"], bocf_results["fpg_change"]),
        ("fpg_change_pct", "FPG reduction (%)", res_fpg_pct, ac_results["fpg_change_pct"], bocf_results["fpg_change_pct"]),
    ]:
        d = r_itt["diff"]
        rows_main.append({
            "Outcome": label, "Analysis": "ITT (MI + ANCOVA, MAR)",
            "N_EPS": N_RAND_E, "N_Human": N_RAND_H,
            "EPS Mean": fmt(r_itt["eps_mean"]["est"]),
            "Human Mean": fmt(r_itt["hum_mean"]["est"]),
            "ANCOVA Adj Diff (95% CI)": fmt_ci(d["est"], d["lo"], d["hi"]),
            "P value": fmt_p(d["p"]), "FMI": fmt(d["fmi"], 2),
        })
        rows_main.append({
            "Outcome": label, "Analysis": "Available-case (ANCOVA)",
            "N_EPS": ac["n_eps"], "N_Human": ac["n_hum"],
            "EPS Mean": fmt(ac["eps_mean"]),
            "Human Mean": fmt(ac["hum_mean"]),
            "ANCOVA Adj Diff (95% CI)": fmt_ci(ac["diff"], ac["lo"], ac["hi"]),
            "P value": fmt_p(ac["p"]), "FMI": "",
        })
        rows_main.append({
            "Outcome": label, "Analysis": "BOCF (missing = baseline FPG)",
            "N_EPS": N_RAND_E, "N_Human": N_RAND_H,
            "EPS Mean": "", "Human Mean": "",
            "ANCOVA Adj Diff (95% CI)": fmt_ci(bocf["diff"], bocf["lo"], bocf["hi"]),
            "P value": fmt_p(bocf["p"]), "FMI": "",
        })

    df_main = pd.DataFrame(rows_main)

    rows_delta = []
    for delta in delta_values:
        d = delta_results[delta]["diff"]
        rows_delta.append({
            "Delta on endline FPG (mmol/L, EPS dropouts)": f"+{delta:.1f}",
            "ANCOVA Adj Diff (95% CI)": fmt_ci(d["est"], d["lo"], d["hi"]),
            "P value": fmt_p(d["p"]),
            "Interpretation": "MAR" if delta == 0 else f"EPS dropouts' endline FPG {delta:.1f} mmol/L higher (worse) than MAR",
        })
    df_delta = pd.DataFrame(rows_delta)
    df_diag  = pd.DataFrame(diag_rows)

    notes = pd.DataFrame([
        {"Item": "Positioning",        "Details": "EXPLORATORY supplementary sensitivity analysis"},
        {"Item": "Randomized",         "Details": f"{N_RAND_H}+{N_RAND_E}={N_RAND_H + N_RAND_E}"},
        {"Item": "Completers",         "Details": f"Human={len(comp_h)}, EPS={len(comp_e)}"},
        {"Item": "Limitation",         "Details": "Missing baselines resampled from completers"},
        {"Item": "Imputation target",  "Details": "Endline FPG ONLY; change scores derived deterministically"},
        {"Item": "MI method",          "Details": "MICE + BayesianRidge, sample_posterior=True"},
        {"Item": "Analysis model",     "Details": "ANCOVA: fpg_change ~ group + fpg0 + age + sex + bmi"},
        {"Item": "MNAR delta",         "Details": "Shift on endline FPG for EPS dropouts (positive = higher glucose = worse)"},
        {"Item": "BOCF",               "Details": "Missing endline FPG set to baseline FPG (0 change)"},
        {"Item": "Seed",               "Details": str(SEED)},
        {"Item": "Caution",            "Details": "Small sample; results should not be over-interpreted"},
    ])

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df_main.to_excel(writer,  sheet_name="Main Results", index=False)
        df_delta.to_excel(writer, sheet_name="EPS Delta Sensitivity", index=False)
        df_diag.to_excel(writer,  sheet_name="MI Diagnostics", index=False)
        notes.to_excel(writer,    sheet_name="Notes", index=False)

    print(f"\n✅ Saved to: {out_xlsx}")


if __name__ == "__main__":
    main()
