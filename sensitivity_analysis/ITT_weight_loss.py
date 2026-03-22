"""
ITT (Intention-to-Treat) Sensitivity Analysis - Weight-Loss Cohort
==================================================================
Positioning: This is a sensitivity analysis supplementing the
available-case primary analysis reported in the manuscript.

Default workflow:
  1. Build the ITT population from completers plus synthetic missing
     baselines resampled within each arm
  2. Perform MAR multiple imputation on weight-loss kg only
  3. Re-derive weight-loss percent deterministically
  4. Pool ANCOVA and modified-Poisson models via Rubin's rules
  5. Report available-case ANCOVA for comparison
  6. Report BOCF (missing = 0 change)
  7. Summarize MI diagnostics

Optional extension:
  If `--weight_human_missing` / `--weight_eps_missing` are supplied,
  those files are used as the missing-participant baseline records
  instead of within-arm resampling. This keeps the current repository's
  example-first defaults while allowing the real-missing-data workflow
  from the updated analyses.

Note: MNAR delta-adjustment and tipping-point analyses are handled in
`tipping_point_analysis.py` so that missing-data sensitivity is defined in
one place only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean_sex(series):
    raw = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    out[raw.isin({"女", "female", "f", "0", "2"})] = 1.0
    out[raw.isin({"男", "male", "m", "1"})] = 0.0
    return out


def pick_col(df, candidates):
    mapping = {str(col).strip().lower(): str(col).strip() for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in mapping:
            return mapping[key]
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt(value, digits=2):
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def fmt_ci(est, lo, hi, digits=2):
    if any(pd.isna(x) for x in [est, lo, hi]):
        return ""
    return f"{est:.{digits}f} ({lo:.{digits}f}, {hi:.{digits}f})"


def fmt_p(value):
    if pd.isna(value):
        return ""
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def describe_source(source):
    if source == "resampled_from_completers":
        return "Within-arm resampling from completers"
    if source.startswith("file:"):
        return f"Missing baseline file: {source[5:]}"
    return source


def load_weight_arm(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_baseline_wt = pick_col(
        df,
        ["baseline_weight_kg", "入营体重", "入营体重kg", "入营体重（档案）", "初始体重（档案）"],
    )
    col_hba1c = pick_col(df, ["hba1c", "HbA1c", "糖化血红蛋白"])
    col_weight_loss = pick_col(df, ["weight_loss_kg", "减重数"])

    arm = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "baseline_wt": clean_numeric(df[col_baseline_wt]).values if col_baseline_wt else np.nan,
            "hba1c": clean_numeric(df[col_hba1c]).values if col_hba1c else np.nan,
            "wl_kg": clean_numeric(df[col_weight_loss]).values if col_weight_loss else np.nan,
        }
    )
    arm["group"] = label
    arm["completer"] = 1
    arm["wl_pct"] = np.where(arm["baseline_wt"] > 0, arm["wl_kg"] / arm["baseline_wt"] * 100, np.nan)
    return arm


def load_weight_missing(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_baseline_wt = pick_col(
        df,
        ["baseline_weight_kg", "入营体重", "入营体重kg", "入营体重（档案）", "初始体重（档案）"],
    )
    col_hba1c = pick_col(df, ["hba1c", "HbA1c", "糖化血红蛋白"])

    hba1c = clean_numeric(df[col_hba1c]) if col_hba1c else pd.Series(np.nan, index=df.index, dtype=float)
    hba1c = hba1c.copy()
    hba1c[hba1c == 0] = np.nan

    missing = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "baseline_wt": clean_numeric(df[col_baseline_wt]).values if col_baseline_wt else np.nan,
            "hba1c": hba1c.values,
            "wl_kg": np.nan,
            "wl_pct": np.nan,
        }
    )
    missing["group"] = label
    missing["completer"] = 0
    return missing


BASELINE_COLS = ["group", "age", "sex", "bmi", "height", "baseline_wt", "hba1c"]


def generate_missing_baseline(comp_df, n_missing, rng):
    if n_missing <= 0:
        return pd.DataFrame(columns=comp_df.columns)

    baseline_cols = [col for col in BASELINE_COLS if col in comp_df.columns]
    sampled = comp_df[baseline_cols].sample(n=n_missing, replace=True, random_state=rng).reset_index(drop=True)
    for col in ["age", "bmi", "height", "baseline_wt", "hba1c"]:
        if col not in sampled.columns:
            continue
        valid = sampled[col].notna()
        if not valid.any():
            continue
        sd = comp_df[col].std() * 0.1
        if pd.notna(sd) and sd > 0:
            sampled[col] = sampled[col].astype(float)
            sampled.loc[valid, col] += rng.normal(0, sd, size=valid.sum())

    sampled["wl_kg"] = np.nan
    sampled["wl_pct"] = np.nan
    sampled["completer"] = 0
    return sampled


def resolve_missing_baseline(comp_df, label, n_expected, rng, missing_path=None):
    if missing_path:
        return load_weight_missing(missing_path, label), f"file:{missing_path}"
    return generate_missing_baseline(comp_df, n_expected, rng), "resampled_from_completers"


def run_mice(df_full, m=20, seed=42):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    candidate_covars = ["group_num", "age", "sex", "bmi", "height", "baseline_wt", "hba1c"]
    mi_covars = [col for col in candidate_covars if col in df_full.columns and df_full[col].notna().any()]
    impute_cols = mi_covars + ["wl_kg"]

    base = df_full[impute_cols].copy()
    datasets = []
    for index in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=50,
            random_state=seed + index,
            sample_posterior=True,
            skip_complete=True,
        )
        filled = pd.DataFrame(imputer.fit_transform(base.values), columns=impute_cols)
        dataset = df_full.copy()
        dataset["wl_kg"] = filled["wl_kg"]
        baseline_wt = filled["baseline_wt"] if "baseline_wt" in filled.columns else dataset["baseline_wt"]
        dataset["wl_pct"] = np.where(baseline_wt > 0, dataset["wl_kg"] / baseline_wt * 100, np.nan)
        datasets.append(dataset)
    return datasets


def rubins_rules(estimates, variances):
    from scipy.stats import t as tdist

    est = np.array(estimates, dtype=float)
    var = np.array(variances, dtype=float)
    valid = ~(np.isnan(est) | np.isnan(var))
    est, var = est[valid], var[valid]

    m = len(est)
    if m < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    q_bar = np.mean(est)
    u_bar = np.mean(var)
    b = np.var(est, ddof=1)
    total_var = u_bar + (1 + 1 / m) * b
    se = math.sqrt(total_var) if total_var > 0 else np.nan
    fmi = ((1 + 1 / m) * b / total_var) if total_var > 0 else np.nan

    if b > 0 and u_bar > 0:
        r = (1 + 1 / m) * b / u_bar
        df = (m - 1) * (1 + 1 / r) ** 2
    else:
        df = 1e6

    tcrit = float(tdist.ppf(0.975, df))
    ci_lo = q_bar - tcrit * se if not np.isnan(se) else np.nan
    ci_hi = q_bar + tcrit * se if not np.isnan(se) else np.nan
    p_value = 2 * float(tdist.sf(abs(q_bar / se), df)) if not np.isnan(se) and se > 0 else np.nan
    return q_bar, se, ci_lo, ci_hi, p_value, df, fmi


def ancova_group_effect(df_c, outcome_col, covars=None):
    import statsmodels.api as sm

    if covars is None:
        covars = ["baseline_wt", "age", "sex", "bmi"]
    covars = [col for col in covars if col in df_c.columns]

    cols = ["group_num"] + covars + [outcome_col]
    data = df_c[cols].dropna()
    if len(data) < len(cols) + 5:
        return np.nan, np.nan, np.nan, np.nan

    y = data[outcome_col].astype(float)
    x = sm.add_constant(data[["group_num"] + covars].astype(float))
    try:
        model = sm.OLS(y, x).fit()
        coef = float(model.params["group_num"])
        se = float(model.bse["group_num"])
        p_value = float(model.pvalues["group_num"])
        return coef, se, se**2, p_value
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def modified_poisson_rr(df_c, responder_col, covars=None):
    import statsmodels.api as sm

    if covars is None:
        covars = ["baseline_wt", "age", "sex", "bmi"]
    covars = [col for col in covars if col in df_c.columns]

    cols = ["group_num"] + covars + [responder_col]
    data = df_c[cols].dropna()
    if len(data) < len(cols) + 5 or data[responder_col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan

    y = data[responder_col].astype(float)
    x = sm.add_constant(data[["group_num"] + covars].astype(float))
    try:
        model = sm.GLM(y, x, family=sm.families.Poisson()).fit(cov_type="HC1")
        log_rr = float(model.params["group_num"])
        se = float(model.bse["group_num"])
        p_value = float(model.pvalues["group_num"])
        return log_rr, se, se**2, p_value
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def pool_ancova(datasets, outcome_col, covars=None):
    coefs, variances = [], []
    eps_means, eps_vars = [], []
    human_means, human_vars = [], []

    for dataset in datasets:
        coef, se, variance, _ = ancova_group_effect(dataset, outcome_col, covars)
        coefs.append(coef)
        variances.append(variance)

        eps_values = dataset.loc[dataset["group"] == "EPS", outcome_col].dropna()
        human_values = dataset.loc[dataset["group"] == "Human", outcome_col].dropna()

        eps_means.append(eps_values.mean())
        eps_vars.append((eps_values.std(ddof=1) / math.sqrt(len(eps_values))) ** 2 if len(eps_values) > 1 else np.nan)
        human_means.append(human_values.mean())
        human_vars.append(
            (human_values.std(ddof=1) / math.sqrt(len(human_values))) ** 2 if len(human_values) > 1 else np.nan
        )

    q_bar, se, lo, hi, p_value, df, fmi = rubins_rules(coefs, variances)
    eps_q, _, eps_lo, eps_hi, _, _, _ = rubins_rules(eps_means, eps_vars)
    human_q, _, human_lo, human_hi, _, _, _ = rubins_rules(human_means, human_vars)
    return {
        "diff": {"est": q_bar, "se": se, "lo": lo, "hi": hi, "p": p_value, "fmi": fmi, "df": df},
        "eps_mean": {"est": eps_q, "lo": eps_lo, "hi": eps_hi},
        "hum_mean": {"est": human_q, "lo": human_lo, "hi": human_hi},
    }


def pool_responder(datasets, threshold_pct, covars=None):
    log_rrs, variances = [], []
    eps_props, human_props = [], []

    for dataset in datasets:
        working = dataset.copy()
        working["_resp"] = (working["wl_pct"] >= threshold_pct).astype(float)
        log_rr, se, variance, _ = modified_poisson_rr(working, "_resp", covars)
        log_rrs.append(log_rr)
        variances.append(variance)

        eps_values = working.loc[working["group"] == "EPS", "_resp"].dropna()
        human_values = working.loc[working["group"] == "Human", "_resp"].dropna()
        eps_props.append(eps_values.mean() * 100 if len(eps_values) > 0 else np.nan)
        human_props.append(human_values.mean() * 100 if len(human_values) > 0 else np.nan)

    q_bar, se, lo, hi, p_value, _, fmi = rubins_rules(log_rrs, variances)
    rr = math.exp(q_bar) if not np.isnan(q_bar) else np.nan
    rr_lo = math.exp(lo) if not np.isnan(lo) else np.nan
    rr_hi = math.exp(hi) if not np.isnan(hi) else np.nan
    return {
        "log_rr": q_bar,
        "rr": rr,
        "rr_lo": rr_lo,
        "rr_hi": rr_hi,
        "p": p_value,
        "fmi": fmi,
        "eps_pct": np.nanmean(eps_props) if len(eps_props) > 0 else np.nan,
        "hum_pct": np.nanmean(human_props) if len(human_props) > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="ITT sensitivity analysis for the weight-loss cohort")
    parser.add_argument(
        "--weight_human",
        type=str,
        default="data/example/weight_loss/human_arm.xlsx",
        help="Path to the Human-arm weight-loss Excel file",
    )
    parser.add_argument(
        "--weight_eps",
        type=str,
        default="data/example/weight_loss/eps_arm.xlsx",
        help="Path to the EPS-arm weight-loss Excel file",
    )
    parser.add_argument(
        "--weight_human_missing",
        type=str,
        default=None,
        help="Optional Human-arm missing-baseline Excel file; if omitted, missing baselines are resampled",
    )
    parser.add_argument(
        "--weight_eps_missing",
        type=str,
        default=None,
        help="Optional EPS-arm missing-baseline Excel file; if omitted, missing baselines are resampled",
    )
    parser.add_argument(
        "--n_randomized_human",
        type=int,
        default=100,
        help="Number randomized to the Human arm",
    )
    parser.add_argument(
        "--n_randomized_eps",
        type=int,
        default=100,
        help="Number randomized to the EPS arm",
    )
    parser.add_argument("--m_imputations", type=int, default=20, help="Number of multiple imputations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/sensitivity_analysis",
        help="Directory where the Excel output will be written",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_xlsx = out_dir / "ITT_weight_loss_results.xlsx"

    df_comp_human = load_weight_arm(args.weight_human, "Human")
    df_comp_eps = load_weight_arm(args.weight_eps, "EPS")
    print(f"Completers: EPS={len(df_comp_eps)}, Human={len(df_comp_human)}")

    n_miss_human_expected = max(0, args.n_randomized_human - len(df_comp_human))
    n_miss_eps_expected = max(0, args.n_randomized_eps - len(df_comp_eps))
    print(f"Expected missing from randomized counts: EPS={n_miss_eps_expected}, Human={n_miss_human_expected}")

    rng = np.random.default_rng(args.seed)
    df_miss_human, source_human = resolve_missing_baseline(
        df_comp_human,
        "Human",
        n_miss_human_expected,
        rng,
        missing_path=args.weight_human_missing,
    )
    df_miss_eps, source_eps = resolve_missing_baseline(
        df_comp_eps,
        "EPS",
        n_miss_eps_expected,
        rng,
        missing_path=args.weight_eps_missing,
    )

    if args.weight_human_missing and len(df_miss_human) != n_miss_human_expected:
        print(
            f"Warning: Human missing file has {len(df_miss_human)} rows, "
            f"but randomized count implies {n_miss_human_expected} missing participants."
        )
    if args.weight_eps_missing and len(df_miss_eps) != n_miss_eps_expected:
        print(
            f"Warning: EPS missing file has {len(df_miss_eps)} rows, "
            f"but randomized count implies {n_miss_eps_expected} missing participants."
        )

    n_itt_human = len(df_comp_human) + len(df_miss_human)
    n_itt_eps = len(df_comp_eps) + len(df_miss_eps)
    print(f"Missing baselines used: EPS={len(df_miss_eps)}, Human={len(df_miss_human)}")
    print(f"Missing sources: EPS={describe_source(source_eps)} | Human={describe_source(source_human)}")

    df_itt = pd.concat([df_comp_human, df_comp_eps, df_miss_human, df_miss_eps], ignore_index=True)
    df_itt["group_num"] = (df_itt["group"] == "EPS").astype(float)

    print(f"ITT: N={len(df_itt)} (Human={sum(df_itt.group == 'Human')}, EPS={sum(df_itt.group == 'EPS')})")
    print(f"Missing wl_kg: {df_itt['wl_kg'].isna().sum()}")

    print("\nRunning MICE (MAR)...")
    imputed_mar = run_mice(df_itt, m=args.m_imputations, seed=args.seed)
    print(f"  {len(imputed_mar)} imputed datasets generated.")

    print("\n=== ITT - ANCOVA (MAR MI) ===")
    res_kg = pool_ancova(imputed_mar, "wl_kg")
    res_pct = pool_ancova(imputed_mar, "wl_pct")
    res_ge2 = pool_responder(imputed_mar, 2)
    res_ge5 = pool_responder(imputed_mar, 5)

    for label, result in [("wl_kg", res_kg), ("wl_pct", res_pct)]:
        diff = result["diff"]
        print(
            f"  {label}: ANCOVA diff = {fmt_ci(diff['est'], diff['lo'], diff['hi'])}, "
            f"p={fmt_p(diff['p'])}, FMI={fmt(diff['fmi'])}"
        )
    print(f"  >=2% RR: {fmt_ci(res_ge2['rr'], res_ge2['rr_lo'], res_ge2['rr_hi'])}, p={fmt_p(res_ge2['p'])}")
    print(f"  >=5% RR: {fmt_ci(res_ge5['rr'], res_ge5['rr_lo'], res_ge5['rr_hi'])}, p={fmt_p(res_ge5['p'])}")

    print("\n=== Available-case - ANCOVA ===")
    from scipy.stats import t as tdist

    df_ac = pd.concat([df_comp_human, df_comp_eps], ignore_index=True)
    df_ac["group_num"] = (df_ac["group"] == "EPS").astype(float)

    ac_results = {}
    for outcome in ["wl_kg", "wl_pct"]:
        coef, se, _, p_value = ancova_group_effect(df_ac, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_ac) - 6, 1)))
        ac_results[outcome] = {
            "diff": coef,
            "se": se,
            "lo": coef - tcrit * se if not np.isnan(se) else np.nan,
            "hi": coef + tcrit * se if not np.isnan(se) else np.nan,
            "p": p_value,
            "n_eps": len(df_comp_eps),
            "n_hum": len(df_comp_human),
            "eps_mean": df_comp_eps[outcome].mean(),
            "hum_mean": df_comp_human[outcome].mean(),
        }
        print(f"  {outcome}: diff={fmt_ci(coef, ac_results[outcome]['lo'], ac_results[outcome]['hi'])}, p={fmt_p(p_value)}")

    print("\n=== BOCF (missing = 0 change) ===")
    df_bocf = df_itt.copy()
    missing_mask = df_bocf["completer"] == 0
    df_bocf.loc[missing_mask, "wl_kg"] = 0.0
    df_bocf.loc[missing_mask, "wl_pct"] = 0.0

    bocf_results = {}
    for outcome in ["wl_kg", "wl_pct"]:
        coef, se, _, p_value = ancova_group_effect(df_bocf, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_bocf) - 6, 1)))
        bocf_results[outcome] = {
            "diff": coef,
            "lo": coef - tcrit * se if not np.isnan(se) else np.nan,
            "hi": coef + tcrit * se if not np.isnan(se) else np.nan,
            "p": p_value,
        }
        print(f"  {outcome}: diff={fmt_ci(coef, bocf_results[outcome]['lo'], bocf_results[outcome]['hi'])}, p={fmt_p(p_value)}")

    print("\n=== MI Diagnostics ===")
    observed_wl = df_itt.loc[df_itt["completer"] == 1, "wl_kg"].dropna()
    imputed_values = []
    for dataset in imputed_mar:
        imputed_values.extend(dataset.loc[df_itt["completer"] == 0, "wl_kg"].dropna().values)
    imputed_wl = np.array(imputed_values)

    diag_rows = [
        {"Metric": "Observed wl_kg - N", "Value": len(observed_wl)},
        {"Metric": "Observed wl_kg - Mean", "Value": f"{observed_wl.mean():.3f}"},
        {"Metric": "Observed wl_kg - SD", "Value": f"{observed_wl.std():.3f}"},
        {"Metric": "Observed wl_kg - Min", "Value": f"{observed_wl.min():.3f}"},
        {"Metric": "Observed wl_kg - Max", "Value": f"{observed_wl.max():.3f}"},
        {"Metric": "Imputed wl_kg - N", "Value": f"{len(imputed_wl)} (across {args.m_imputations} imputations)"},
        {"Metric": "Imputed wl_kg - Mean", "Value": f"{imputed_wl.mean():.3f}" if len(imputed_wl) > 0 else "N/A"},
        {"Metric": "Imputed wl_kg - SD", "Value": f"{imputed_wl.std():.3f}" if len(imputed_wl) > 0 else "N/A"},
        {"Metric": "Imputed wl_kg - Min", "Value": f"{imputed_wl.min():.3f}" if len(imputed_wl) > 0 else "N/A"},
        {"Metric": "Imputed wl_kg - Max", "Value": f"{imputed_wl.max():.3f}" if len(imputed_wl) > 0 else "N/A"},
        {"Metric": "FMI (wl_kg ANCOVA)", "Value": fmt(res_kg["diff"]["fmi"])},
        {"Metric": "FMI (wl_pct ANCOVA)", "Value": fmt(res_pct["diff"]["fmi"])},
        {"Metric": "FMI (>=2% mod-Poisson)", "Value": fmt(res_ge2["fmi"])},
    ]
    print(f"  Observed wl_kg: mean={observed_wl.mean():.3f}, sd={observed_wl.std():.3f}")
    if len(imputed_wl) > 0:
        print(f"  Imputed  wl_kg: mean={imputed_wl.mean():.3f}, sd={imputed_wl.std():.3f}")
    print(f"  FMI: wl_kg={fmt(res_kg['diff']['fmi'])}, wl_pct={fmt(res_pct['diff']['fmi'])}")

    rows_main = []
    for outcome, label, itt_result, ac_result in [
        ("wl_kg", "Weight loss (kg)", res_kg, ac_results["wl_kg"]),
        ("wl_pct", "Weight loss (%)", res_pct, ac_results["wl_pct"]),
    ]:
        diff = itt_result["diff"]
        rows_main.append(
            {
                "Outcome": label,
                "Analysis": "ITT (MI + ANCOVA, MAR)",
                "N_EPS": n_itt_eps,
                "N_Human": n_itt_human,
                "EPS Mean (95% CI)": fmt_ci(
                    itt_result["eps_mean"]["est"], itt_result["eps_mean"]["lo"], itt_result["eps_mean"]["hi"]
                ),
                "Human Mean (95% CI)": fmt_ci(
                    itt_result["hum_mean"]["est"], itt_result["hum_mean"]["lo"], itt_result["hum_mean"]["hi"]
                ),
                "ANCOVA Adj Diff (95% CI)": fmt_ci(diff["est"], diff["lo"], diff["hi"]),
                "P value": fmt_p(diff["p"]),
                "FMI": fmt(diff["fmi"]),
            }
        )
        rows_main.append(
            {
                "Outcome": label,
                "Analysis": "Available-case (ANCOVA)",
                "N_EPS": ac_result["n_eps"],
                "N_Human": ac_result["n_hum"],
                "EPS Mean (95% CI)": fmt(ac_result["eps_mean"]),
                "Human Mean (95% CI)": fmt(ac_result["hum_mean"]),
                "ANCOVA Adj Diff (95% CI)": fmt_ci(ac_result["diff"], ac_result["lo"], ac_result["hi"]),
                "P value": fmt_p(ac_result["p"]),
                "FMI": "",
            }
        )
        bocf = bocf_results[outcome]
        rows_main.append(
            {
                "Outcome": label,
                "Analysis": "BOCF (missing = 0 change, ANCOVA)",
                "N_EPS": n_itt_eps,
                "N_Human": n_itt_human,
                "EPS Mean (95% CI)": "",
                "Human Mean (95% CI)": "",
                "ANCOVA Adj Diff (95% CI)": fmt_ci(bocf["diff"], bocf["lo"], bocf["hi"]),
                "P value": fmt_p(bocf["p"]),
                "FMI": "",
            }
        )

    for threshold, result in [(2, res_ge2), (5, res_ge5)]:
        rows_main.append(
            {
                "Outcome": f">={threshold}% responder",
                "Analysis": "ITT (MI + mod-Poisson, MAR)",
                "N_EPS": n_itt_eps,
                "N_Human": n_itt_human,
                "EPS Mean (95% CI)": f"{result['eps_pct']:.1f}%",
                "Human Mean (95% CI)": f"{result['hum_pct']:.1f}%",
                "ANCOVA Adj Diff (95% CI)": f"RR {fmt_ci(result['rr'], result['rr_lo'], result['rr_hi'])}",
                "P value": fmt_p(result["p"]),
                "FMI": fmt(result["fmi"]),
            }
        )

    df_main = pd.DataFrame(rows_main)
    df_diag = pd.DataFrame(diag_rows)

    limitation = (
        "Baseline covariates for missing participants are approximated via within-arm resampling from completers"
        if "resampled_from_completers" in {source_human, source_eps}
        else "Missing participants use supplied baseline records from the missing-data files"
    )
    notes = pd.DataFrame(
        [
            {"Item": "Primary analysis", "Details": "Available-case analysis reported in the manuscript"},
            {"Item": "This script", "Details": "Sensitivity analysis: ITT with MI under MAR"},
            {"Item": "Randomized target", "Details": f"Human={args.n_randomized_human}, EPS={args.n_randomized_eps}"},
            {"Item": "ITT N used", "Details": f"Human={n_itt_human}, EPS={n_itt_eps}"},
            {"Item": "Missing baseline source (Human)", "Details": describe_source(source_human)},
            {"Item": "Missing baseline source (EPS)", "Details": describe_source(source_eps)},
            {"Item": "Limitation", "Details": limitation},
            {
                "Item": "Imputation target",
                "Details": "Weight-loss kg only; weight-loss percent is re-derived deterministically from baseline weight",
            },
            {"Item": "MI method", "Details": "MICE via IterativeImputer + BayesianRidge, sample_posterior=True"},
            {
                "Item": "Analysis model (continuous)",
                "Details": "ANCOVA: outcome ~ group + baseline_wt + age + sex + bmi",
            },
            {
                "Item": "Analysis model (binary)",
                "Details": "Modified Poisson with robust SE: responder ~ group + baseline_wt + age + sex + bmi",
            },
            {"Item": "Pooling", "Details": "Rubin's rules with Barnard-Rubin df adjustment"},
            {"Item": "BOCF", "Details": "Missing participants assigned 0 weight change"},
            {
                "Item": "MNAR / tipping point",
                "Details": "See tipping_point_analysis.py for delta-adjustment and tipping-point sensitivity analyses",
            },
            {"Item": "Seed", "Details": str(args.seed)},
        ]
    )

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="Main Results", index=False)
        df_diag.to_excel(writer, sheet_name="MI Diagnostics", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"\nSaved to: {out_xlsx}")


if __name__ == "__main__":
    main()
