"""
ITT Sensitivity Analysis - Weight-Loss Cohort
=============================================
Positioning: Primary ITT sensitivity analysis for the weight-loss cohort.

Reference workflow:
  1. Load completer data from the randomized cohort.
  2. If missing-baseline Excel files are supplied, use those real baseline
     records for non-completers.
  3. Repository fallback: if missing-baseline files are not supplied,
     reconstruct the missing participants by within-arm resampling from
     completers so the bundled example data still run end-to-end.
  4. Build the ITT population and impute weight-loss kg under MAR.
  5. Re-derive weight-loss percent deterministically.
  6. Pool ANCOVA and modified-Poisson models via Rubin's rules.
  7. Report available-case ANCOVA, BOCF, and MI diagnostics.

Note: MNAR delta-adjustment and tipping-point analyses are handled in
`tipping_point_analysis.py` so that missing-data sensitivity is defined in
one place only.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*", category=RuntimeWarning)


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


def fmt(value, digits=3):
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def fmt_ci(est, lo, hi, digits=3):
    if any(pd.isna(x) for x in [est, lo, hi]):
        return ""
    return f"{est:.{digits}f} ({lo:.{digits}f}, {hi:.{digits}f})"


def fmt_p(value):
    if pd.isna(value):
        return ""
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def describe_source(source):
    if source == "resampled_from_completers":
        return "Within-arm resampling from completers (repository fallback)"
    if source.startswith("file:"):
        return f"Missing baseline file: {source[5:]}"
    return source


def nanmean_or_nan(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


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
IMPUTE_COLS = ["group_num", "age", "sex", "bmi", "height", "baseline_wt", "hba1c", "wl_kg"]


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

    mi_covars = [col for col in IMPUTE_COLS[:-1] if col in df_full.columns and df_full[col].notna().any()]
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


def ancova_effect(df_c, outcome_col, covars=None):
    import statsmodels.api as sm

    if covars is None:
        covars = ["baseline_wt", "age", "sex", "bmi"]
    covars = [col for col in covars if col in df_c.columns]

    cols = ["group_num"] + covars + [outcome_col]
    data = df_c[cols].dropna()
    if len(data) < len(cols) + 3:
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


def add_responder_cols(datasets):
    out = []
    for dataset in datasets:
        dataset_copy = dataset.copy()
        dataset_copy["resp_2pct"] = np.where(
            dataset_copy["wl_pct"].isna(), np.nan, (dataset_copy["wl_pct"] >= 2).astype(float)
        )
        dataset_copy["resp_5pct"] = np.where(
            dataset_copy["wl_pct"].isna(), np.nan, (dataset_copy["wl_pct"] >= 5).astype(float)
        )
        out.append(dataset_copy)
    return out


def modified_poisson_effect(df_c, outcome_col, covars=None):
    import statsmodels.api as sm

    if covars is None:
        covars = ["baseline_wt", "age", "sex", "bmi"]
    covars = [col for col in covars if col in df_c.columns]

    cols = ["group_num"] + covars + [outcome_col]
    data = df_c[cols].dropna()
    if len(data) < len(cols) + 3 or data[outcome_col].sum() < 3:
        return np.nan, np.nan

    y = data[outcome_col].astype(float)
    x = sm.add_constant(data[["group_num"] + covars].astype(float))
    try:
        model = sm.GLM(y, x, family=sm.families.Poisson()).fit(cov_type="HC1")
        log_rr = float(model.params["group_num"])
        var_log = float(model.cov_params().loc["group_num", "group_num"])
        return log_rr, var_log
    except Exception:
        return np.nan, np.nan


def pool_ancova(datasets, outcome_col, covars=None):
    coefs, variances = [], []
    eps_means, eps_vars = [], []
    human_means, human_vars = [], []

    for dataset in datasets:
        coef, se, variance, _ = ancova_effect(dataset, outcome_col, covars)
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

    q_bar, se, lo, hi, p_value, _, fmi = rubins_rules(coefs, variances)
    eps_q, _, eps_lo, eps_hi, _, _, _ = rubins_rules(eps_means, eps_vars)
    human_q, _, human_lo, human_hi, _, _, _ = rubins_rules(human_means, human_vars)
    return {
        "diff": {"est": q_bar, "se": se, "lo": lo, "hi": hi, "p": p_value, "fmi": fmi},
        "eps_mean": {"est": eps_q, "lo": eps_lo, "hi": eps_hi},
        "hum_mean": {"est": human_q, "lo": human_lo, "hi": human_hi},
    }


def pool_poisson(datasets, outcome_col, covars=None):
    log_rrs, variances = [], []
    eps_props, human_props = [], []

    for dataset in datasets:
        log_rr, variance = modified_poisson_effect(dataset, outcome_col, covars)
        log_rrs.append(log_rr)
        variances.append(variance)
        eps_props.append(dataset.loc[dataset["group"] == "EPS", outcome_col].mean())
        human_props.append(dataset.loc[dataset["group"] == "Human", outcome_col].mean())

    q_bar, se, lo, hi, p_value, _, fmi = rubins_rules(log_rrs, variances)
    rr = math.exp(q_bar) if not np.isnan(q_bar) else np.nan
    rr_lo = math.exp(lo) if not np.isnan(lo) else np.nan
    rr_hi = math.exp(hi) if not np.isnan(hi) else np.nan
    return {
        "rr": rr,
        "rr_lo": rr_lo,
        "rr_hi": rr_hi,
        "p": p_value,
        "fmi": fmi,
        "eps_prop": nanmean_or_nan(eps_props),
        "hum_prop": nanmean_or_nan(human_props),
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
        help="Optional Human-arm missing-baseline Excel file",
    )
    parser.add_argument(
        "--weight_eps_missing",
        type=str,
        default=None,
        help="Optional EPS-arm missing-baseline Excel file",
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

    comp_h = load_weight_arm(args.weight_human, "Human")
    comp_e = load_weight_arm(args.weight_eps, "EPS")
    print(f"Completers: Human={len(comp_h)}, EPS={len(comp_e)}")

    n_miss_h_expected = max(0, args.n_randomized_human - len(comp_h))
    n_miss_e_expected = max(0, args.n_randomized_eps - len(comp_e))
    print(f"Expected missing from randomized counts: Human={n_miss_h_expected}, EPS={n_miss_e_expected}")

    rng = np.random.default_rng(args.seed)
    miss_h, source_h = resolve_missing_baseline(
        comp_h,
        "Human",
        n_miss_h_expected,
        rng,
        missing_path=args.weight_human_missing,
    )
    miss_e, source_e = resolve_missing_baseline(
        comp_e,
        "EPS",
        n_miss_e_expected,
        rng,
        missing_path=args.weight_eps_missing,
    )

    if args.weight_human_missing and len(miss_h) != n_miss_h_expected:
        print(
            f"Warning: Human missing file has {len(miss_h)} rows, "
            f"but randomized count implies {n_miss_h_expected} missing participants."
        )
    if args.weight_eps_missing and len(miss_e) != n_miss_e_expected:
        print(
            f"Warning: EPS missing file has {len(miss_e)} rows, "
            f"but randomized count implies {n_miss_e_expected} missing participants."
        )

    for arm, n_comp, n_miss, n_rand in [
        ("Human", len(comp_h), len(miss_h), args.n_randomized_human),
        ("EPS", len(comp_e), len(miss_e), args.n_randomized_eps),
    ]:
        total = n_comp + n_miss
        mark = "OK" if total == n_rand else "WARNING"
        print(f"{mark}: {arm} {n_comp}+{n_miss}={total} (target={n_rand})")

    n_itt_h = len(comp_h) + len(miss_h)
    n_itt_e = len(comp_e) + len(miss_e)
    print(f"Missing baselines used: Human={len(miss_h)}, EPS={len(miss_e)}")
    print(f"Missing sources: Human={describe_source(source_h)} | EPS={describe_source(source_e)}")

    df_itt = pd.concat([comp_h, comp_e, miss_h, miss_e], ignore_index=True)
    df_itt["group_num"] = (df_itt["group"] == "EPS").astype(float)

    print(f"ITT: N={len(df_itt)} (Human={sum(df_itt.group == 'Human')}, EPS={sum(df_itt.group == 'EPS')})")
    print(f"Missing wl_kg: {df_itt['wl_kg'].isna().sum()}")
    print(
        f"Completer rate: Human={sum((df_itt.group == 'Human') & (df_itt.completer == 1))}/{sum(df_itt.group == 'Human')}, "
        f"EPS={sum((df_itt.group == 'EPS') & (df_itt.completer == 1))}/{sum(df_itt.group == 'EPS')}"
    )

    print("\nRunning MICE (MAR)...")
    imputed_mar = run_mice(df_itt, m=args.m_imputations, seed=args.seed)
    print(f"  {len(imputed_mar)} imputed datasets.")

    print("\n=== ITT - ANCOVA (MAR MI) ===")
    res_kg = pool_ancova(imputed_mar, "wl_kg")
    res_pct = pool_ancova(imputed_mar, "wl_pct")
    for label, result in [("WL (kg)", res_kg), ("WL (%)", res_pct)]:
        diff = result["diff"]
        print(
            f"  {label}: diff={fmt_ci(diff['est'], diff['lo'], diff['hi'])}, "
            f"p={fmt_p(diff['p'])}, FMI={fmt(diff['fmi'], 2)}"
        )

    print("\n=== Available-case - ANCOVA ===")
    from scipy.stats import t as tdist

    df_ac = pd.concat([comp_h, comp_e], ignore_index=True)
    df_ac["group_num"] = (df_ac["group"] == "EPS").astype(float)

    ac_results = {}
    for outcome, label in [("wl_kg", "WL (kg)"), ("wl_pct", "WL (%)")]:
        coef, se, _, p_value = ancova_effect(df_ac, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_ac) - 6, 1)))
        ac_results[outcome] = {
            "diff": coef,
            "lo": coef - tcrit * se if not np.isnan(se) else np.nan,
            "hi": coef + tcrit * se if not np.isnan(se) else np.nan,
            "p": p_value,
            "n_eps": len(comp_e),
            "n_hum": len(comp_h),
            "eps_mean": comp_e[outcome].mean(),
            "hum_mean": comp_h[outcome].mean(),
        }
        print(f"  {label}: diff={fmt_ci(coef, ac_results[outcome]['lo'], ac_results[outcome]['hi'])}, p={fmt_p(p_value)}")

    print("\n=== BOCF (missing = 0 kg weight loss) ===")
    df_bocf = df_itt.copy()
    missing_mask = df_bocf["completer"] == 0
    df_bocf.loc[missing_mask, "wl_kg"] = 0.0
    df_bocf.loc[missing_mask, "wl_pct"] = 0.0

    bocf_results = {}
    for outcome in ["wl_kg", "wl_pct"]:
        coef, se, _, p_value = ancova_effect(df_bocf, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_bocf) - 6, 1)))
        bocf_results[outcome] = {
            "diff": coef,
            "lo": coef - tcrit * se if not np.isnan(se) else np.nan,
            "hi": coef + tcrit * se if not np.isnan(se) else np.nan,
            "p": p_value,
        }
        print(f"  {outcome}: diff={fmt_ci(coef, bocf_results[outcome]['lo'], bocf_results[outcome]['hi'])}, p={fmt_p(p_value)}")

    print("\n=== Modified Poisson - Binary Responders ===")
    imputed_resp = add_responder_cols(imputed_mar)
    res_2pct = pool_poisson(imputed_resp, "resp_2pct")
    res_5pct = pool_poisson(imputed_resp, "resp_5pct")
    for label, result in [(">=2% WL responders", res_2pct), (">=5% WL responders", res_5pct)]:
        rr_ci = fmt_ci(result["rr"], result["rr_lo"], result["rr_hi"])
        print(f"  {label}: RR={rr_ci}, p={fmt_p(result['p'])}, FMI={fmt(result['fmi'], 2)}")
        print(f"    EPS prop={fmt(result['eps_prop'], 3)}, Human prop={fmt(result['hum_prop'], 3)}")

    observed_wl = df_itt.loc[df_itt["completer"] == 1, "wl_kg"].dropna()
    imputed_values = []
    for dataset in imputed_mar:
        imputed_values.extend(dataset.loc[df_itt["completer"] == 0, "wl_kg"].dropna().values)
    imputed_wl = np.array(imputed_values)

    diag_rows = [
        {"Metric": "Observed wl_kg - N", "Value": len(observed_wl)},
        {"Metric": "Observed wl_kg - Mean", "Value": f"{observed_wl.mean():.3f}"},
        {"Metric": "Observed wl_kg - SD", "Value": f"{observed_wl.std():.3f}"},
        {"Metric": "Imputed wl_kg - N", "Value": f"{len(imputed_wl)} (across {args.m_imputations} imputations)"},
        {"Metric": "Imputed wl_kg - Mean", "Value": f"{imputed_wl.mean():.3f}" if len(imputed_wl) > 0 else "N/A"},
        {"Metric": "Imputed wl_kg - SD", "Value": f"{imputed_wl.std():.3f}" if len(imputed_wl) > 0 else "N/A"},
        {"Metric": "FMI (wl_kg ANCOVA)", "Value": fmt(res_kg["diff"]["fmi"], 2)},
        {"Metric": "FMI (wl_pct ANCOVA)", "Value": fmt(res_pct["diff"]["fmi"], 2)},
    ]

    rows_main = []
    for outcome, label, itt_result, ac_result, bocf_result in [
        ("wl_kg", "Weight loss (kg)", res_kg, ac_results["wl_kg"], bocf_results["wl_kg"]),
        ("wl_pct", "Weight loss (%)", res_pct, ac_results["wl_pct"], bocf_results["wl_pct"]),
    ]:
        diff = itt_result["diff"]
        rows_main.append(
            {
                "Outcome": label,
                "Analysis": "ITT (MI + ANCOVA, MAR)",
                "N_EPS": n_itt_e,
                "N_Human": n_itt_h,
                "EPS Mean": fmt(itt_result["eps_mean"]["est"]),
                "Human Mean": fmt(itt_result["hum_mean"]["est"]),
                "ANCOVA Adj Diff (95% CI)": fmt_ci(diff["est"], diff["lo"], diff["hi"]),
                "P value": fmt_p(diff["p"]),
                "FMI": fmt(diff["fmi"], 2),
            }
        )
        rows_main.append(
            {
                "Outcome": label,
                "Analysis": "Available-case (ANCOVA)",
                "N_EPS": ac_result["n_eps"],
                "N_Human": ac_result["n_hum"],
                "EPS Mean": fmt(ac_result["eps_mean"]),
                "Human Mean": fmt(ac_result["hum_mean"]),
                "ANCOVA Adj Diff (95% CI)": fmt_ci(ac_result["diff"], ac_result["lo"], ac_result["hi"]),
                "P value": fmt_p(ac_result["p"]),
                "FMI": "",
            }
        )
        rows_main.append(
            {
                "Outcome": label,
                "Analysis": "BOCF (missing = 0 kg loss)",
                "N_EPS": n_itt_e,
                "N_Human": n_itt_h,
                "EPS Mean": "",
                "Human Mean": "",
                "ANCOVA Adj Diff (95% CI)": fmt_ci(bocf_result["diff"], bocf_result["lo"], bocf_result["hi"]),
                "P value": fmt_p(bocf_result["p"]),
                "FMI": "",
            }
        )

    rows_poisson = []
    for label, result in [(">=2% weight loss", res_2pct), (">=5% weight loss", res_5pct)]:
        rows_poisson.append(
            {
                "Outcome": label,
                "Analysis": "ITT (Modified Poisson, MAR MI)",
                "N_EPS": n_itt_e,
                "N_Human": n_itt_h,
                "EPS Prop": fmt(result["eps_prop"], 3),
                "Human Prop": fmt(result["hum_prop"], 3),
                "RR (95% CI)": (
                    f"{result['rr']:.3f} ({result['rr_lo']:.3f}, {result['rr_hi']:.3f})"
                    if not any(np.isnan(x) for x in [result["rr"], result["rr_lo"], result["rr_hi"]])
                    else ""
                ),
                "P value": fmt_p(result["p"]),
                "FMI": fmt(result["fmi"], 2),
            }
        )

    df_main = pd.DataFrame(rows_main)
    df_poisson = pd.DataFrame(rows_poisson)
    df_diag = pd.DataFrame(diag_rows)

    if "resampled_from_completers" in {source_h, source_e}:
        missing_workflow = (
            "Missing-baseline files were not supplied, so the repository fallback "
            "reconstructed missing participants by within-arm resampling from completers."
        )
    else:
        missing_workflow = "Supplied missing-baseline files were used as the real baseline records for non-completers."

    notes = pd.DataFrame(
        [
            {"Item": "Positioning", "Details": "Primary ITT sensitivity analysis - weight-loss cohort"},
            {"Item": "Randomized target", "Details": f"Human={args.n_randomized_human}, EPS={args.n_randomized_eps}"},
            {"Item": "Completers", "Details": f"Human={len(comp_h)}, EPS={len(comp_e)}"},
            {"Item": "ITT N used", "Details": f"Human={n_itt_h}, EPS={n_itt_e}"},
            {"Item": "Missing baseline source (Human)", "Details": describe_source(source_h)},
            {"Item": "Missing baseline source (EPS)", "Details": describe_source(source_e)},
            {"Item": "Missing-baseline workflow", "Details": missing_workflow},
            {"Item": "HbA1c note", "Details": "HbA1c=0 in missing-baseline files is treated as not measured (NaN)"},
            {"Item": "Imputation target", "Details": "wl_kg; wl_pct is re-derived from wl_kg / baseline_wt"},
            {"Item": "MI covariates", "Details": ", ".join(IMPUTE_COLS)},
            {"Item": "MI method", "Details": "MICE via IterativeImputer + BayesianRidge, sample_posterior=True"},
            {
                "Item": "Analysis model (continuous)",
                "Details": "ANCOVA: outcome ~ group + baseline_wt + age + sex + bmi",
            },
            {
                "Item": "Analysis model (binary)",
                "Details": "Modified Poisson with robust SE: responder ~ group + baseline_wt + age + sex + bmi",
            },
            {"Item": "BOCF", "Details": "Missing participants assigned 0 kg / 0% weight loss"},
            {"Item": "MNAR / tipping point", "Details": "See tipping_point_analysis.py"},
            {"Item": "Seed", "Details": str(args.seed)},
        ]
    )

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="Main Results", index=False)
        df_poisson.to_excel(writer, sheet_name="Responder Analysis", index=False)
        df_diag.to_excel(writer, sheet_name="MI Diagnostics", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"\nSaved to: {out_xlsx}")


if __name__ == "__main__":
    main()
