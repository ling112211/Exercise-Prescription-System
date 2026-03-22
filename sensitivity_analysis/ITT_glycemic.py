"""
ITT Sensitivity Analysis - Glycemic-Control Cohort
=================================================
Positioning: This exploratory supplementary analysis extends the
available-case primary analysis for the glycemic-control cohort.

Default workflow:
  1. Construct the ITT population from completers plus synthetic missing
     baselines resampled within each arm
  2. Perform MAR multiple imputation on endline fasting glucose only
  3. Re-derive fasting-glucose change metrics deterministically
  4. Pool ANCOVA models via Rubin's rules
  5. Report available-case ANCOVA for comparison
  6. Report BOCF (missing endline FPG = baseline FPG)
  7. Summarize MI diagnostics

Optional extension:
  If `--gly_human_missing` / `--gly_eps_missing` are supplied, those
  baseline files are used for the missing participants instead of
  within-arm resampling. This preserves the current repository defaults
  while supporting the updated real-missing-data workflow.

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
        return "Within-arm resampling from completers"
    if source.startswith("file:"):
        return f"Missing baseline file: {source[5:]}"
    return source


def load_glycemic_arm(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_bw = pick_col(df, ["baseline_weight_kg", "入营体重", "入营体重kg", "初始体重（档案）"])
    col_fpg0 = pick_col(df, ["baseline_fpg_mmol", "入营空腹"])
    col_ppg0 = pick_col(df, ["baseline_ppg_mmol", "入营餐后2小时"])
    col_fpg1 = pick_col(df, ["endpoint_fpg_mmol", "结营空腹"])
    col_ppg1 = pick_col(df, ["endpoint_ppg_mmol", "结营餐后2小时"])

    arm = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "bw": clean_numeric(df[col_bw]).values if col_bw else np.nan,
            "fpg0": clean_numeric(df[col_fpg0]).values if col_fpg0 else np.nan,
            "ppg0": clean_numeric(df[col_ppg0]).values if col_ppg0 else np.nan,
            "fpg1": clean_numeric(df[col_fpg1]).values if col_fpg1 else np.nan,
            "ppg1": clean_numeric(df[col_ppg1]).values if col_ppg1 else np.nan,
        }
    )
    arm["group"] = label
    arm["completer"] = 1
    arm["fpg_change"] = arm["fpg0"] - arm["fpg1"]
    arm["fpg_change_pct"] = np.where(arm["fpg0"] > 0, arm["fpg_change"] / arm["fpg0"] * 100, np.nan)
    return arm


def build_glycemic_missing_frame(df, label):
    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_bw = pick_col(df, ["baseline_weight_kg", "入营体重", "入营体重kg", "初始体重（档案）"])
    col_fpg0 = pick_col(df, ["baseline_fpg_mmol", "入营空腹"])
    col_ppg0 = pick_col(df, ["baseline_ppg_mmol", "入营餐后2小时"])

    missing = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "bw": clean_numeric(df[col_bw]).values if col_bw else np.nan,
            "fpg0": clean_numeric(df[col_fpg0]).values if col_fpg0 else np.nan,
            "ppg0": clean_numeric(df[col_ppg0]).values if col_ppg0 else np.nan,
            "fpg1": np.nan,
            "ppg1": np.nan,
            "fpg_change": np.nan,
            "fpg_change_pct": np.nan,
        }
    )
    missing["group"] = label
    missing["completer"] = 0
    return missing


def eps_missing_needs_shift_correction(df):
    col_age = pick_col(df, ["年龄"])
    if not col_age:
        return False
    observed = df[col_age]
    non_null = observed.notna()
    if non_null.sum() == 0:
        return False
    numeric = clean_numeric(observed[non_null]).notna().sum()
    return numeric < non_null.sum()


def load_shifted_eps_missing(path):
    df_raw = pd.read_excel(path, header=None)
    headers = df_raw.iloc[0].tolist()

    col_meas = headers.index("最新测量数据时间")
    col_age = headers.index("年龄")
    col_sex = headers.index("性别")
    col_ht = headers.index("身高")
    col_wt = headers.index("初始体重（档案）")
    col_bmi = headers.index("bmi")
    col_fpg0 = headers.index("入营空腹")

    records = []
    for _, row in df_raw.iloc[1:].iterrows():
        age_val = row.iloc[col_age]
        try:
            age = float(age_val)
            sex_raw = str(row.iloc[col_sex])
            height = float(row.iloc[col_ht]) if pd.notna(row.iloc[col_ht]) else np.nan
            weight = float(row.iloc[col_wt]) if pd.notna(row.iloc[col_wt]) else np.nan
            bmi = float(row.iloc[col_bmi]) if pd.notna(row.iloc[col_bmi]) else np.nan
        except (TypeError, ValueError):
            age = float(row.iloc[col_meas]) if pd.notna(row.iloc[col_meas]) else np.nan
            sex_raw = str(row.iloc[col_age])
            height = float(row.iloc[col_sex]) if pd.notna(row.iloc[col_sex]) else np.nan
            weight = float(row.iloc[col_ht]) if pd.notna(row.iloc[col_ht]) else np.nan
            bmi = float(row.iloc[col_wt]) if pd.notna(row.iloc[col_wt]) else np.nan

        fpg0 = float(row.iloc[col_fpg0]) if pd.notna(row.iloc[col_fpg0]) else np.nan
        records.append({"age": age, "sex_raw": sex_raw, "height": height, "bw": weight, "bmi": bmi, "fpg0": fpg0})

    missing = pd.DataFrame(records)
    missing["sex"] = clean_sex(missing["sex_raw"].astype(str))
    missing["ppg0"] = np.nan
    missing["fpg1"] = np.nan
    missing["ppg1"] = np.nan
    missing["fpg_change"] = np.nan
    missing["fpg_change_pct"] = np.nan
    missing["group"] = "EPS"
    missing["completer"] = 0
    return missing[["age", "sex", "bmi", "height", "bw", "fpg0", "ppg0", "fpg1", "ppg1", "group", "completer", "fpg_change", "fpg_change_pct"]]


def load_glycemic_missing(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    if label == "EPS" and eps_missing_needs_shift_correction(df):
        return load_shifted_eps_missing(path), "Applied EPS one-column shift correction to demographic fields"

    return build_glycemic_missing_frame(df, label), ""


BASELINE_COLS = ["group", "age", "sex", "bmi", "height", "bw", "fpg0", "ppg0"]


def generate_missing(comp_df, n_missing, rng):
    if n_missing <= 0:
        return pd.DataFrame(columns=comp_df.columns)

    baseline_cols = [col for col in BASELINE_COLS if col in comp_df.columns]
    sampled = comp_df[baseline_cols].sample(n=n_missing, replace=True, random_state=rng).reset_index(drop=True)
    for col in ["age", "bmi", "height", "bw", "fpg0", "ppg0"]:
        if col not in sampled.columns:
            continue
        valid = sampled[col].notna()
        if not valid.any():
            continue
        sd = comp_df[col].std() * 0.1
        if pd.notna(sd) and sd > 0:
            sampled[col] = sampled[col].astype(float)
            sampled.loc[valid, col] += rng.normal(0, sd, size=valid.sum())

    sampled["fpg1"] = np.nan
    sampled["ppg1"] = np.nan
    sampled["fpg_change"] = np.nan
    sampled["fpg_change_pct"] = np.nan
    sampled["completer"] = 0
    return sampled


def resolve_missing_baseline(comp_df, label, n_expected, rng, missing_path=None):
    if missing_path:
        missing, note = load_glycemic_missing(missing_path, label)
        return missing, f"file:{missing_path}", note
    return generate_missing(comp_df, n_expected, rng), "resampled_from_completers", ""


def run_mice(df_full, m=20, seed=42):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    candidate_covars = ["group_num", "age", "sex", "bmi", "fpg0", "ppg0"]
    mi_covars = [col for col in candidate_covars if col in df_full.columns and df_full[col].notna().any()]
    impute_cols = mi_covars + ["fpg1"]

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
        dataset["fpg1"] = filled["fpg1"]
        dataset["fpg_change"] = dataset["fpg0"] - dataset["fpg1"]
        dataset["fpg_change_pct"] = np.where(
            dataset["fpg0"] > 0, dataset["fpg_change"] / dataset["fpg0"] * 100, np.nan
        )
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
        covars = ["fpg0", "age", "sex", "bmi"]
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

    q_bar, se, lo, hi, p_value, df, fmi = rubins_rules(coefs, variances)
    eps_q, _, eps_lo, eps_hi, _, _, _ = rubins_rules(eps_means, eps_vars)
    human_q, _, human_lo, human_hi, _, _, _ = rubins_rules(human_means, human_vars)
    return {
        "diff": {"est": q_bar, "se": se, "lo": lo, "hi": hi, "p": p_value, "fmi": fmi, "df": df},
        "eps_mean": {"est": eps_q, "lo": eps_lo, "hi": eps_hi},
        "hum_mean": {"est": human_q, "lo": human_lo, "hi": human_hi},
    }


def main():
    parser = argparse.ArgumentParser(description="ITT sensitivity analysis for the glycemic-control cohort")
    parser.add_argument(
        "--gly_human",
        type=str,
        default="data/example/glycemic/human_arm.xlsx",
        help="Path to the Human-arm glycemic Excel file",
    )
    parser.add_argument(
        "--gly_eps",
        type=str,
        default="data/example/glycemic/eps_arm.xlsx",
        help="Path to the EPS-arm glycemic Excel file",
    )
    parser.add_argument(
        "--gly_human_missing",
        type=str,
        default=None,
        help="Optional Human-arm missing-baseline Excel file; if omitted, missing baselines are resampled",
    )
    parser.add_argument(
        "--gly_eps_missing",
        type=str,
        default=None,
        help="Optional EPS-arm missing-baseline Excel file; if omitted, missing baselines are resampled",
    )
    parser.add_argument(
        "--n_randomized_human",
        type=int,
        default=50,
        help="Number randomized to the Human arm",
    )
    parser.add_argument(
        "--n_randomized_eps",
        type=int,
        default=50,
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
    out_xlsx = out_dir / "ITT_glycemic_results.xlsx"

    comp_h = load_glycemic_arm(args.gly_human, "Human")
    comp_e = load_glycemic_arm(args.gly_eps, "EPS")
    print(f"Completers: Human={len(comp_h)}, EPS={len(comp_e)}")

    n_miss_h_expected = max(0, args.n_randomized_human - len(comp_h))
    n_miss_e_expected = max(0, args.n_randomized_eps - len(comp_e))
    print(f"Expected missing from randomized counts: Human={n_miss_h_expected}, EPS={n_miss_e_expected}")

    rng = np.random.default_rng(args.seed)
    miss_h, source_h, note_h = resolve_missing_baseline(
        comp_h,
        "Human",
        n_miss_h_expected,
        rng,
        missing_path=args.gly_human_missing,
    )
    miss_e, source_e, note_e = resolve_missing_baseline(
        comp_e,
        "EPS",
        n_miss_e_expected,
        rng,
        missing_path=args.gly_eps_missing,
    )

    if args.gly_human_missing and len(miss_h) != n_miss_h_expected:
        print(
            f"Warning: Human missing file has {len(miss_h)} rows, "
            f"but randomized count implies {n_miss_h_expected} missing participants."
        )
    if args.gly_eps_missing and len(miss_e) != n_miss_e_expected:
        print(
            f"Warning: EPS missing file has {len(miss_e)} rows, "
            f"but randomized count implies {n_miss_e_expected} missing participants."
        )

    n_itt_h = len(comp_h) + len(miss_h)
    n_itt_e = len(comp_e) + len(miss_e)
    print(f"Missing baselines used: Human={len(miss_h)}, EPS={len(miss_e)}")
    print(f"Missing sources: Human={describe_source(source_h)} | EPS={describe_source(source_e)}")
    if note_h:
        print(f"Human missing-data note: {note_h}")
    if note_e:
        print(f"EPS missing-data note: {note_e}")

    df_itt = pd.concat([comp_h, comp_e, miss_h, miss_e], ignore_index=True)
    df_itt["group_num"] = (df_itt["group"] == "EPS").astype(float)

    print(f"ITT: N={len(df_itt)} (Human={sum(df_itt.group == 'Human')}, EPS={sum(df_itt.group == 'EPS')})")
    print(f"Missing fpg1: {df_itt['fpg1'].isna().sum()}")

    print("\nRunning MICE (MAR)...")
    imputed_mar = run_mice(df_itt, m=args.m_imputations, seed=args.seed)
    print(f"  {len(imputed_mar)} imputed datasets.")

    print("\n=== ITT - ANCOVA (MAR MI) ===")
    res_fpg = pool_ancova(imputed_mar, "fpg_change")
    res_fpg_pct = pool_ancova(imputed_mar, "fpg_change_pct")

    for label, result in [("FPG change (mmol/L)", res_fpg), ("FPG change (%)", res_fpg_pct)]:
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
    for outcome, label in [("fpg_change", "FPG change"), ("fpg_change_pct", "FPG change %")]:
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

    print("\n=== BOCF (missing = 0 FPG change) ===")
    df_bocf = df_itt.copy()
    missing_mask = df_bocf["completer"] == 0
    df_bocf.loc[missing_mask, "fpg1"] = df_bocf.loc[missing_mask, "fpg0"]
    df_bocf["fpg_change"] = df_bocf["fpg0"] - df_bocf["fpg1"]
    df_bocf["fpg_change_pct"] = np.where(df_bocf["fpg0"] > 0, df_bocf["fpg_change"] / df_bocf["fpg0"] * 100, np.nan)

    bocf_results = {}
    for outcome in ["fpg_change", "fpg_change_pct"]:
        coef, se, _, p_value = ancova_effect(df_bocf, outcome)
        tcrit = float(tdist.ppf(0.975, max(len(df_bocf) - 6, 1)))
        bocf_results[outcome] = {
            "diff": coef,
            "lo": coef - tcrit * se if not np.isnan(se) else np.nan,
            "hi": coef + tcrit * se if not np.isnan(se) else np.nan,
            "p": p_value,
        }
        print(f"  {outcome}: diff={fmt_ci(coef, bocf_results[outcome]['lo'], bocf_results[outcome]['hi'])}, p={fmt_p(p_value)}")

    observed_fpg1 = df_itt.loc[df_itt["completer"] == 1, "fpg1"].dropna()
    imputed_values = []
    for dataset in imputed_mar:
        imputed_values.extend(dataset.loc[df_itt["completer"] == 0, "fpg1"].dropna().values)
    imputed_fpg1 = np.array(imputed_values)

    diag_rows = [
        {"Metric": "Observed endline FPG - N", "Value": len(observed_fpg1)},
        {"Metric": "Observed endline FPG - Mean", "Value": f"{observed_fpg1.mean():.3f}"},
        {"Metric": "Observed endline FPG - SD", "Value": f"{observed_fpg1.std():.3f}"},
        {
            "Metric": "Imputed endline FPG - N",
            "Value": f"{len(imputed_fpg1)} (across {args.m_imputations} imputations)",
        },
        {"Metric": "Imputed endline FPG - Mean", "Value": f"{imputed_fpg1.mean():.3f}" if len(imputed_fpg1) > 0 else "N/A"},
        {"Metric": "Imputed endline FPG - SD", "Value": f"{imputed_fpg1.std():.3f}" if len(imputed_fpg1) > 0 else "N/A"},
        {"Metric": "FMI (fpg_change ANCOVA)", "Value": fmt(res_fpg["diff"]["fmi"], 2)},
        {"Metric": "FMI (fpg_change_pct ANCOVA)", "Value": fmt(res_fpg_pct["diff"]["fmi"], 2)},
    ]

    rows_main = []
    for outcome, label, itt_result, ac_result, bocf_result in [
        ("fpg_change", "FPG reduction (mmol/L)", res_fpg, ac_results["fpg_change"], bocf_results["fpg_change"]),
        (
            "fpg_change_pct",
            "FPG reduction (%)",
            res_fpg_pct,
            ac_results["fpg_change_pct"],
            bocf_results["fpg_change_pct"],
        ),
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
                "Analysis": "BOCF (missing = baseline FPG)",
                "N_EPS": n_itt_e,
                "N_Human": n_itt_h,
                "EPS Mean": "",
                "Human Mean": "",
                "ANCOVA Adj Diff (95% CI)": fmt_ci(bocf_result["diff"], bocf_result["lo"], bocf_result["hi"]),
                "P value": fmt_p(bocf_result["p"]),
                "FMI": "",
            }
        )

    df_main = pd.DataFrame(rows_main)
    df_diag = pd.DataFrame(diag_rows)

    limitation = (
        "Baseline covariates for missing participants are approximated via within-arm resampling from completers"
        if "resampled_from_completers" in {source_h, source_e}
        else "Missing participants use supplied baseline records from the missing-data files"
    )
    notes_rows = [
        {"Item": "Positioning", "Details": "Exploratory supplementary sensitivity analysis"},
        {"Item": "Randomized target", "Details": f"Human={args.n_randomized_human}, EPS={args.n_randomized_eps}"},
        {"Item": "ITT N used", "Details": f"Human={n_itt_h}, EPS={n_itt_e}"},
        {"Item": "Missing baseline source (Human)", "Details": describe_source(source_h)},
        {"Item": "Missing baseline source (EPS)", "Details": describe_source(source_e)},
        {"Item": "Limitation", "Details": limitation},
        {
            "Item": "Imputation target",
            "Details": "Endline fasting glucose only; change metrics are re-derived deterministically",
        },
        {"Item": "MI method", "Details": "MICE via IterativeImputer + BayesianRidge, sample_posterior=True"},
        {"Item": "Analysis model", "Details": "ANCOVA: outcome ~ group + fpg0 + age + sex + bmi"},
        {"Item": "BOCF", "Details": "Missing endline FPG set equal to baseline FPG (0 change)"},
        {
            "Item": "MNAR / tipping point",
            "Details": "See tipping_point_analysis.py for delta-adjustment and tipping-point sensitivity analyses",
        },
        {"Item": "Seed", "Details": str(args.seed)},
        {"Item": "Caution", "Details": "Interpret glycemic ITT results cautiously when the cohort is small"},
    ]
    if note_h:
        notes_rows.append({"Item": "Human missing-data preprocessing", "Details": note_h})
    if note_e:
        notes_rows.append({"Item": "EPS missing-data preprocessing", "Details": note_e})
    if miss_h["ppg0"].isna().all() and len(miss_h) > 0:
        notes_rows.append({"Item": "Human ppg0 for missing", "Details": "Not available in missing-data file; set to NaN"})
    if miss_e["ppg0"].isna().all() and len(miss_e) > 0:
        notes_rows.append({"Item": "EPS ppg0 for missing", "Details": "Not available in missing-data file; set to NaN"})
    notes = pd.DataFrame(notes_rows)

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="Main Results", index=False)
        df_diag.to_excel(writer, sheet_name="MI Diagnostics", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"\nSaved to: {out_xlsx}")


if __name__ == "__main__":
    main()
