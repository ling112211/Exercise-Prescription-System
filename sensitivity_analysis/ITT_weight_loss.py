"""
ITT (Intention-to-Treat) Sensitivity Analysis — Weight Loss Cohort
===================================================================
Positioning:  This is a SENSITIVITY ANALYSIS supplementing the available-case
              primary analysis reported in the manuscript.

Population:   N_RANDOMIZED per arm (default 100 for example data); completers
              are those present in the input files.
Limitation:   Full baseline CRF for non-completers was unavailable at time of
              analysis. Missing participants' baseline covariates are therefore
              resampled (with jitter) from the completer distribution. This is
              a known limitation and should be stated in the manuscript.

Workflow:
  1. Construct ITT population (completers + bootstrapped missing baselines)
  2. MICE imputation of wl_kg ONLY (MAR) → derive wl_pct, responder
  3. ANCOVA on each imputed dataset → pool via Rubin's rules
  4. Modified Poisson for binary responder → pool via Rubin's rules
  5. MNAR delta-adjustment (arm-specific) on EPS dropouts
  6. BOCF worst-case
  7. MI diagnostics (FMI, distribution summary)

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
    """Map sex labels to numeric: female=1, male=0."""
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


def fmt(v, d=2):
    return "" if pd.isna(v) else f"{v:.{d}f}"

def fmt_ci(e, lo, hi, d=2):
    return "" if any(pd.isna(x) for x in [e, lo, hi]) else f"{e:.{d}f} ({lo:.{d}f}, {hi:.{d}f})"

def fmt_p(p):
    return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")


# ============================================================
# 1) Load & clean available-case data
# ============================================================
def load_arm(path, label):
    """Load one arm's data from an Excel file with flexible column names."""
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["BMI", "bmi"])
    col_bw  = pick_col(df, ["baseline_weight_kg", "入营体重", "入营体重kg"])
    col_ht  = pick_col(df, ["height", "身高"])
    col_hba1c = pick_col(df, ["hba1c", "HbA1c", "糖化血红蛋白"])
    col_wl  = pick_col(df, ["weight_loss_kg", "减重数"])

    d = pd.DataFrame({
        "age":         clean_numeric(df[col_age]).values if col_age else np.nan,
        "sex":         clean_sex(df[col_sex]).values if col_sex else np.nan,
        "bmi":         clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
        "height":      clean_numeric(df[col_ht]).values if col_ht else np.nan,
        "baseline_wt": clean_numeric(df[col_bw]).values if col_bw else np.nan,
        "hba1c":       clean_numeric(df[col_hba1c]).values if col_hba1c else np.nan,
        "wl_kg":       clean_numeric(df[col_wl]).values if col_wl else np.nan,
    })
    d["group"]     = label
    d["completer"] = 1
    d["wl_pct"] = np.where(d["baseline_wt"] > 0,
                           d["wl_kg"] / d["baseline_wt"] * 100, np.nan)
    return d


# ============================================================
# 2) Construct ITT population — bootstrap missing baselines
# ============================================================
BASELINE_COLS = ["group", "age", "sex", "bmi", "height", "baseline_wt", "hba1c"]


def generate_missing_baseline(comp_df, n_missing, rng):
    """Resample baseline covariates from completers for missing participants."""
    if n_missing <= 0:
        return pd.DataFrame(columns=comp_df.columns)
    available_bl = [c for c in BASELINE_COLS if c in comp_df.columns]
    sampled = comp_df[available_bl].sample(
        n=n_missing, replace=True, random_state=rng
    ).reset_index(drop=True)
    for col in ["age", "bmi", "height", "baseline_wt", "hba1c"]:
        if col not in sampled.columns:
            continue
        v = sampled[col].notna()
        if v.any():
            sd = comp_df[col].std() * 0.1
            if pd.notna(sd) and sd > 0:
                sampled[col] = sampled[col].astype(float)
                sampled.loc[v, col] += rng.normal(0, sd, size=v.sum())
    sampled["wl_kg"]     = np.nan
    sampled["wl_pct"]    = np.nan
    sampled["completer"] = 0
    return sampled


# ============================================================
# 3) MICE — impute wl_kg ONLY, derive wl_pct deterministically
# ============================================================
def run_mice(df_full, m=20, seed=42):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.linear_model import BayesianRidge
    from sklearn.impute import IterativeImputer

    # Build imputation feature set from available columns
    candidate_covars = ["group_num", "age", "sex", "bmi", "height", "baseline_wt", "hba1c"]
    mi_covars = [c for c in candidate_covars if c in df_full.columns and df_full[c].notna().any()]
    impute_cols = mi_covars + ["wl_kg"]

    base = df_full[impute_cols].copy()
    datasets = []
    for i in range(m):
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=50,
            random_state=seed + i,
            sample_posterior=True,
            skip_complete=True,
        )
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=impute_cols)
        dc = df_full.copy()
        dc["wl_kg"] = filled["wl_kg"]
        bw = filled["baseline_wt"] if "baseline_wt" in filled.columns else dc["baseline_wt"]
        dc["wl_pct"] = np.where(bw > 0, dc["wl_kg"] / bw * 100, np.nan)
        datasets.append(dc)
    return datasets


# ============================================================
# 4) Rubin's rules
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

    Q = np.mean(est)
    U = np.mean(var)
    B = np.var(est, ddof=1)
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


# ============================================================
# 5) Analysis functions — ANCOVA + modified Poisson
# ============================================================
def ancova_group_effect(df_c, outcome_col, covars=None):
    import statsmodels.api as sm
    if covars is None:
        covars = ["baseline_wt", "age", "sex", "bmi"]
    covars = [c for c in covars if c in df_c.columns]
    cols = ["group_num"] + covars + [outcome_col]
    d = df_c[cols].dropna()
    if len(d) < len(cols) + 5:
        return np.nan, np.nan, np.nan, np.nan
    y = d[outcome_col].astype(float)
    X = sm.add_constant(d[["group_num"] + covars].astype(float))
    try:
        model = sm.OLS(y, X).fit()
        coef = float(model.params["group_num"])
        se   = float(model.bse["group_num"])
        p    = float(model.pvalues["group_num"])
        return coef, se, se**2, p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def modified_poisson_rr(df_c, responder_col, covars=None):
    import statsmodels.api as sm
    if covars is None:
        covars = ["baseline_wt", "age", "sex", "bmi"]
    covars = [c for c in covars if c in df_c.columns]
    cols = ["group_num"] + covars + [responder_col]
    d = df_c[cols].dropna()
    if len(d) < len(cols) + 5 or d[responder_col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    y = d[responder_col].astype(float)
    X = sm.add_constant(d[["group_num"] + covars].astype(float))
    try:
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
        log_rr = float(model.params["group_num"])
        se     = float(model.bse["group_num"])
        p      = float(model.pvalues["group_num"])
        return log_rr, se, se**2, p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ============================================================
# 6) Pooling helpers
# ============================================================
def pool_ancova(datasets, outcome_col, covars=None):
    coefs, variances = [], []
    eps_means, eps_vars = [], []
    hum_means, hum_vars = [], []
    for dc in datasets:
        c, se, v, p = ancova_group_effect(dc, outcome_col, covars)
        coefs.append(c); variances.append(v)
        e = dc.loc[dc.group == "EPS", outcome_col].dropna()
        h = dc.loc[dc.group == "Human", outcome_col].dropna()
        eps_means.append(e.mean())
        eps_vars.append((e.std(ddof=1)/math.sqrt(len(e)))**2 if len(e) > 1 else np.nan)
        hum_means.append(h.mean())
        hum_vars.append((h.std(ddof=1)/math.sqrt(len(h)))**2 if len(h) > 1 else np.nan)

    Q, se, lo, hi, p, df, fmi = rubins_rules(coefs, variances)
    eQ, _, eLo, eHi, _, _, _ = rubins_rules(eps_means, eps_vars)
    hQ, _, hLo, hHi, _, _, _ = rubins_rules(hum_means, hum_vars)
    return {
        "diff": {"est": Q, "se": se, "lo": lo, "hi": hi, "p": p, "fmi": fmi, "df": df},
        "eps_mean": {"est": eQ, "lo": eLo, "hi": eHi},
        "hum_mean": {"est": hQ, "lo": hLo, "hi": hHi},
    }


def pool_responder(datasets, threshold_pct, covars=None):
    log_rrs, variances = [], []
    eps_props, hum_props = [], []
    for dc in datasets:
        dc["_resp"] = (dc["wl_pct"] >= threshold_pct).astype(float)
        lr, se, v, p = modified_poisson_rr(dc, "_resp", covars)
        log_rrs.append(lr); variances.append(v)
        e = dc.loc[dc.group == "EPS", "_resp"].dropna()
        h = dc.loc[dc.group == "Human", "_resp"].dropna()
        eps_props.append(e.mean() * 100 if len(e) > 0 else np.nan)
        hum_props.append(h.mean() * 100 if len(h) > 0 else np.nan)

    Q, se, lo, hi, p, df, fmi = rubins_rules(log_rrs, variances)
    rr    = math.exp(Q) if not np.isnan(Q) else np.nan
    rr_lo = math.exp(lo) if not np.isnan(lo) else np.nan
    rr_hi = math.exp(hi) if not np.isnan(hi) else np.nan
    return {
        "log_rr": Q, "rr": rr, "rr_lo": rr_lo, "rr_hi": rr_hi,
        "p": p, "fmi": fmi,
        "eps_pct": np.mean(eps_props), "hum_pct": np.mean(hum_props),
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="ITT sensitivity analysis — weight-loss cohort")
    parser.add_argument("--weight_human", type=str,
                        default="data/example/weight_loss/human_arm.xlsx",
                        help="Path to Human-arm weight-loss Excel file")
    parser.add_argument("--weight_eps", type=str,
                        default="data/example/weight_loss/eps_arm.xlsx",
                        help="Path to EPS-arm weight-loss Excel file")
    parser.add_argument("--n_randomized_human", type=int, default=100,
                        help="Number randomized in Human arm (default 100 for example data)")
    parser.add_argument("--n_randomized_eps", type=int, default=100,
                        help="Number randomized in EPS arm (default 100 for example data)")
    parser.add_argument("--m_imputations", type=int, default=20,
                        help="Number of MI datasets (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default="outputs/sensitivity_analysis",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_xlsx = out_dir / "ITT_weight_loss_results.xlsx"

    M = args.m_imputations
    SEED = args.seed
    N_RAND_H = args.n_randomized_human
    N_RAND_E = args.n_randomized_eps

    # --- Load ---
    df_comp_eps   = load_arm(args.weight_eps,   "EPS")
    df_comp_human = load_arm(args.weight_human, "Human")
    print(f"Completers: EPS={len(df_comp_eps)}, Human={len(df_comp_human)}")

    # --- ITT population ---
    n_miss_human = max(0, N_RAND_H - len(df_comp_human))
    n_miss_eps   = max(0, N_RAND_E - len(df_comp_eps))
    print(f"Missing: EPS={n_miss_eps}, Human={n_miss_human}")

    rng = np.random.default_rng(SEED)
    df_itt = pd.concat([
        df_comp_human, df_comp_eps,
        generate_missing_baseline(df_comp_human, n_miss_human, rng),
        generate_missing_baseline(df_comp_eps,   n_miss_eps,   rng),
    ], ignore_index=True)
    df_itt["group_num"] = (df_itt["group"] == "EPS").astype(float)

    print(f"ITT: N={len(df_itt)} (Human={sum(df_itt.group=='Human')}, EPS={sum(df_itt.group=='EPS')})")
    print(f"Missing wl_kg: {df_itt['wl_kg'].isna().sum()}")

    # --- MICE (MAR) ---
    print("\nRunning MICE (MAR)...")
    imputed_mar = run_mice(df_itt, m=M, seed=SEED)
    print(f"  {len(imputed_mar)} imputed datasets generated.")

    # --- ITT ANCOVA ---
    print("\n=== ITT — ANCOVA (MAR MI) ===")
    res_kg  = pool_ancova(imputed_mar, "wl_kg")
    res_pct = pool_ancova(imputed_mar, "wl_pct")
    res_ge2 = pool_responder(imputed_mar, 2)
    res_ge5 = pool_responder(imputed_mar, 5)

    for label, r in [("wl_kg", res_kg), ("wl_pct", res_pct)]:
        d = r["diff"]
        print(f"  {label}: ANCOVA diff = {fmt_ci(d['est'], d['lo'], d['hi'])}, p={fmt_p(d['p'])}, FMI={fmt(d['fmi'])}")
    print(f"  ≥2% RR: {fmt_ci(res_ge2['rr'], res_ge2['rr_lo'], res_ge2['rr_hi'])}, p={fmt_p(res_ge2['p'])}")
    print(f"  ≥5% RR: {fmt_ci(res_ge5['rr'], res_ge5['rr_lo'], res_ge5['rr_hi'])}, p={fmt_p(res_ge5['p'])}")

    # --- Available-case ANCOVA ---
    print("\n=== Available-case — ANCOVA ===")
    from scipy.stats import t as tdist
    df_ac = pd.concat([df_comp_human, df_comp_eps], ignore_index=True)
    df_ac["group_num"] = (df_ac["group"] == "EPS").astype(float)

    ac_results = {}
    for outcome in ["wl_kg", "wl_pct"]:
        c, se, v, p = ancova_group_effect(df_ac, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_ac) - 6, 1)))
        ac_results[outcome] = {
            "diff": c, "se": se,
            "lo": c - tcrit * se if not np.isnan(se) else np.nan,
            "hi": c + tcrit * se if not np.isnan(se) else np.nan,
            "p": p,
            "n_eps": len(df_comp_eps), "n_hum": len(df_comp_human),
            "eps_mean": df_comp_eps[outcome].mean(),
            "hum_mean": df_comp_human[outcome].mean(),
        }
        print(f"  {outcome}: diff={fmt_ci(c, ac_results[outcome]['lo'], ac_results[outcome]['hi'])}, p={fmt_p(p)}")

    # --- MNAR delta ---
    print("\n=== MNAR Delta-Adjustment (EPS dropouts only) ===")
    delta_values = [0, -0.5, -1.0, -1.5, -2.0, -3.0, -5.0]
    delta_results = {}
    for delta in delta_values:
        shifted = []
        for dc in run_mice(df_itt, m=10, seed=SEED):
            ds = dc.copy()
            eps_miss = (df_itt["completer"] == 0) & (df_itt["group"] == "EPS")
            ds.loc[eps_miss, "wl_kg"] += delta
            bw = ds["baseline_wt"]
            ds["wl_pct"] = np.where(bw > 0, ds["wl_kg"] / bw * 100, np.nan)
            shifted.append(ds)
        r_kg  = pool_ancova(shifted, "wl_kg")
        r_pct = pool_ancova(shifted, "wl_pct")
        delta_results[delta] = {"wl_kg": r_kg, "wl_pct": r_pct}
        d = r_kg["diff"]
        label = "MAR" if delta == 0 else f"delta={delta:.1f}kg"
        print(f"  {label}: wl_kg diff = {fmt_ci(d['est'], d['lo'], d['hi'])}, p={fmt_p(d['p'])}")

    # --- BOCF ---
    print("\n=== BOCF (missing = 0 change) ===")
    df_bocf = pd.concat([df_comp_human, df_comp_eps], ignore_index=True)
    for grp, n_miss in [("Human", n_miss_human), ("EPS", n_miss_eps)]:
        comp = df_comp_human if grp == "Human" else df_comp_eps
        miss = generate_missing_baseline(comp, n_miss, np.random.default_rng(SEED + 1))
        miss["wl_kg"]  = 0.0
        miss["wl_pct"] = 0.0
        df_bocf = pd.concat([df_bocf, miss], ignore_index=True)
    df_bocf["group_num"] = (df_bocf["group"] == "EPS").astype(float)

    bocf_results = {}
    for outcome in ["wl_kg", "wl_pct"]:
        c, se, v, p = ancova_group_effect(df_bocf, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_bocf) - 6, 1)))
        bocf_results[outcome] = {
            "diff": c,
            "lo": c - tcrit * se if not np.isnan(se) else np.nan,
            "hi": c + tcrit * se if not np.isnan(se) else np.nan,
            "p": p,
        }
        print(f"  {outcome}: diff={fmt_ci(c, bocf_results[outcome]['lo'], bocf_results[outcome]['hi'])}, p={fmt_p(p)}")

    # --- MI diagnostics ---
    print("\n=== MI Diagnostics ===")
    obs_wl = df_itt.loc[df_itt.completer == 1, "wl_kg"].dropna()
    imp_wls = []
    for dc in imputed_mar:
        imp_wls.extend(dc.loc[df_itt.completer == 0, "wl_kg"].dropna().values)
    imp_wl = np.array(imp_wls)

    diag_rows = [
        {"Metric": "Observed wl_kg — N",    "Value": len(obs_wl)},
        {"Metric": "Observed wl_kg — Mean",  "Value": f"{obs_wl.mean():.3f}"},
        {"Metric": "Observed wl_kg — SD",    "Value": f"{obs_wl.std():.3f}"},
        {"Metric": "Observed wl_kg — Min",   "Value": f"{obs_wl.min():.3f}"},
        {"Metric": "Observed wl_kg — Max",   "Value": f"{obs_wl.max():.3f}"},
        {"Metric": "Imputed wl_kg — N",      "Value": f"{len(imp_wl)} (across {M} imputations)"},
        {"Metric": "Imputed wl_kg — Mean",   "Value": f"{imp_wl.mean():.3f}" if len(imp_wl) > 0 else "N/A"},
        {"Metric": "Imputed wl_kg — SD",     "Value": f"{imp_wl.std():.3f}" if len(imp_wl) > 0 else "N/A"},
        {"Metric": "Imputed wl_kg — Min",    "Value": f"{imp_wl.min():.3f}" if len(imp_wl) > 0 else "N/A"},
        {"Metric": "Imputed wl_kg — Max",    "Value": f"{imp_wl.max():.3f}" if len(imp_wl) > 0 else "N/A"},
        {"Metric": "FMI (wl_kg ANCOVA)",     "Value": fmt(res_kg["diff"]["fmi"])},
        {"Metric": "FMI (wl_pct ANCOVA)",    "Value": fmt(res_pct["diff"]["fmi"])},
        {"Metric": "FMI (≥2% mod-Poisson)",  "Value": fmt(res_ge2["fmi"])},
    ]
    print(f"  Observed wl_kg: mean={obs_wl.mean():.3f}, sd={obs_wl.std():.3f}")
    if len(imp_wl) > 0:
        print(f"  Imputed  wl_kg: mean={imp_wl.mean():.3f}, sd={imp_wl.std():.3f}")
    print(f"  FMI: wl_kg={fmt(res_kg['diff']['fmi'])}, wl_pct={fmt(res_pct['diff']['fmi'])}")

    # --- Save ---
    rows_main = []
    for outcome, label, res_itt, ac in [
        ("wl_kg", "Weight loss (kg)", res_kg, ac_results["wl_kg"]),
        ("wl_pct", "Weight loss (%)", res_pct, ac_results["wl_pct"]),
    ]:
        d = res_itt["diff"]
        rows_main.append({
            "Outcome": label, "Analysis": "ITT (MI + ANCOVA, MAR)",
            "N_EPS": N_RAND_E, "N_Human": N_RAND_H,
            "EPS Mean (95% CI)": fmt_ci(res_itt["eps_mean"]["est"], res_itt["eps_mean"]["lo"], res_itt["eps_mean"]["hi"]),
            "Human Mean (95% CI)": fmt_ci(res_itt["hum_mean"]["est"], res_itt["hum_mean"]["lo"], res_itt["hum_mean"]["hi"]),
            "ANCOVA Adj Diff (95% CI)": fmt_ci(d["est"], d["lo"], d["hi"]),
            "P value": fmt_p(d["p"]), "FMI": fmt(d["fmi"]),
        })
        rows_main.append({
            "Outcome": label, "Analysis": "Available-case (ANCOVA)",
            "N_EPS": ac["n_eps"], "N_Human": ac["n_hum"],
            "EPS Mean (95% CI)": fmt(ac["eps_mean"]),
            "Human Mean (95% CI)": fmt(ac["hum_mean"]),
            "ANCOVA Adj Diff (95% CI)": fmt_ci(ac["diff"], ac["lo"], ac["hi"]),
            "P value": fmt_p(ac["p"]), "FMI": "",
        })
        b = bocf_results[outcome]
        rows_main.append({
            "Outcome": label, "Analysis": "BOCF (missing=0 change, ANCOVA)",
            "N_EPS": N_RAND_E, "N_Human": N_RAND_H,
            "EPS Mean (95% CI)": "", "Human Mean (95% CI)": "",
            "ANCOVA Adj Diff (95% CI)": fmt_ci(b["diff"], b["lo"], b["hi"]),
            "P value": fmt_p(b["p"]), "FMI": "",
        })

    for thr, r in [(2, res_ge2), (5, res_ge5)]:
        rows_main.append({
            "Outcome": f"≥{thr}% responder",
            "Analysis": "ITT (MI + mod-Poisson, MAR)",
            "N_EPS": N_RAND_E, "N_Human": N_RAND_H,
            "EPS Mean (95% CI)": f"{r['eps_pct']:.1f}%",
            "Human Mean (95% CI)": f"{r['hum_pct']:.1f}%",
            "ANCOVA Adj Diff (95% CI)": f"RR {fmt_ci(r['rr'], r['rr_lo'], r['rr_hi'])}",
            "P value": fmt_p(r["p"]), "FMI": fmt(r["fmi"]),
        })

    df_main = pd.DataFrame(rows_main)

    rows_delta = []
    for delta in delta_values:
        for outcome, label in [("wl_kg", "Weight loss (kg)"), ("wl_pct", "Weight loss (%)")]:
            d = delta_results[delta][outcome]["diff"]
            rows_delta.append({
                "Outcome": label,
                "Delta (kg, EPS dropouts only)": delta,
                "ANCOVA Adj Diff (95% CI)": fmt_ci(d["est"], d["lo"], d["hi"]),
                "P value": fmt_p(d["p"]),
                "Interpretation": "MAR" if delta == 0 else f"EPS dropouts {abs(delta):.1f}kg less than MAR-imputed",
            })
    df_delta = pd.DataFrame(rows_delta)

    df_diag = pd.DataFrame(diag_rows)

    notes = pd.DataFrame([
        {"Item": "Study design",           "Details": f"RCT, {N_RAND_H}+{N_RAND_E} randomized"},
        {"Item": "Primary analysis",       "Details": "Available-case (as reported in manuscript)"},
        {"Item": "This script",            "Details": "Sensitivity analysis: ITT with MI under MAR + MNAR delta-adjustment"},
        {"Item": "Limitation",             "Details": "Missing participants' baselines resampled from completer distribution (real CRF unavailable)"},
        {"Item": "Imputation target",      "Details": "wl_kg ONLY; wl_pct = wl_kg / baseline_wt × 100 (deterministic derivation)"},
        {"Item": "MI method",              "Details": "MICE via sklearn IterativeImputer + BayesianRidge, sample_posterior=True"},
        {"Item": "Number of imputations",  "Details": str(M)},
        {"Item": "Analysis model (continuous)", "Details": "ANCOVA: outcome ~ group + baseline_wt + age + sex + bmi"},
        {"Item": "Analysis model (binary)",    "Details": "Modified Poisson (robust SE): responder ~ group + baseline_wt + age + sex + bmi"},
        {"Item": "Pooling",                "Details": "Rubin's rules with Barnard-Rubin df adjustment"},
        {"Item": "MNAR sensitivity",       "Details": "Delta-adjustment on EPS-arm dropouts only (most conservative for EPS benefit claim)"},
        {"Item": "BOCF",                   "Details": "Missing = 0 weight change (returned to baseline)"},
        {"Item": "Seed",                   "Details": str(SEED)},
    ])

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df_main.to_excel(writer,  sheet_name="Main Results", index=False)
        df_delta.to_excel(writer, sheet_name="EPS Delta Sensitivity", index=False)
        df_diag.to_excel(writer,  sheet_name="MI Diagnostics", index=False)
        notes.to_excel(writer,    sheet_name="Notes", index=False)

    print(f"\n✅ Saved to: {out_xlsx}")


if __name__ == "__main__":
    main()
