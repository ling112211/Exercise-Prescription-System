# Exercise-Prescription-System

This repository provides the code to reproduce the analyses, tables, and figures for the paper: **An LLM-Based Exercise Prescription System for Digital Chronic Disease Management** (under review).

The trial was prospectively registered at the [Chinese Clinical Trial Registry (ChiCTR2600118939)](https://www.chictr.org.cn/hvshowprojectEN.html?id=295346&v=1.0).

---

## Overview

This repository contains the statistical analysis and visualization code for the Exercise Prescription System (EPS), an expert-aligned large language model (LLM)–based system designed to produce safe, personalized, and scalable exercise feedback for digital chronic disease management.

EPS was developed and evaluated through three stages:

1. **Benchmark evaluation**: Four domain-specific medical question-answering benchmarks (CMB, CMExam, MedMCQA, MedQA) assessing the medical knowledge of EPS variants and baseline models.
2. **Expert pilot study**: An ablation-style evaluation by 25 professional health managers rating exercise-feedback outputs across seven dimensions (consensus, correctness, completeness, unbiasedness, clarity, empathy, and actionability).
3. **Randomized controlled trials**: A single-blind RCT in two cohorts — 1,444 weight-loss participants and 40 glycemic-control participants — comparing EPS–human collaboration (EPS-generated feedback reviewed by health managers) against human coaching alone.

Key results:

- EPS improved medical benchmark performance by 20–30 percentage points over base models across all four benchmarks, with Qwen3-14B (EPS) achieving 86.75% on CMB, 90.66% on CMExam, 89.74% on MedMCQA, and 86.30% on MedQA.
- In the weight-loss RCT, the EPS–human arm achieved significantly greater mean weight loss (1.40 kg vs 1.20 kg; *P* = 0.0004) and a higher proportion of participants achieving ≥2% weight loss (62.94% vs 54.27%; *P* = 0.0008).
- In the glycemic-control RCT, the EPS–human arm yielded significantly larger reductions in fasting glucose (0.98 mmol/L vs 0.30 mmol/L; *P* < 0.05).

## Repository Structure

```
Exercise-Prescription-System/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README_data.md                         # Data dictionary and access instructions
│   ├── benchmark_results/
│   │   └── benchmark_accuracy.csv             # Pre-computed benchmark accuracy (REAL DATA — reproduces Fig. 7)
│   ├── expert_pilot/
│   │   ├── base_model.csv                     # Expert ratings for base model (REAL DATA — reproduces Fig. 8)
│   │   ├── eps_without_d2.csv                 # Expert ratings for base model + D1 (REAL DATA — reproduces Fig. 8)
│   │   └── eps.csv                            # Expert ratings for full EPS (REAL DATA — reproduces Fig. 8)
│   └── example/                               # EXAMPLE DATA ONLY — for code verification, not paper results
│       ├── weight_loss/
│       │   ├── human_arm.xlsx                 # Anonymised example data (does NOT reproduce paper Tables/Figs)
│       │   └── eps_arm.xlsx
│       ├── glycemic/
│       │   ├── human_arm.xlsx
│       │   └── eps_arm.xlsx
│       └── questionnaire/
│           ├── human_responses.xlsx
│           └── eps_responses.xlsx
├── D1/                                        # Public exercise and weight-management corpus (86,900 entries)
│   ├── medbooks-18-cot-filtered.json          # Medical textbook CoT Q&A (filtered)
│   ├── medinstruct-52k-filtered.json          # Medical instruction dataset (filtered)
│   ├── chatdoctor-cleaned-filtered.json       # ChatDoctor patient Q&A (cleaned, filtered)
│   ├── medicationqa-filtered.json             # Medication Q&A (filtered)
│   ├── mts-dialog-filtered.json               # Medical dialogue (filtered)
│   ├── liveqa-filtered.json                   # LiveQA medical Q&A (filtered)
│   ├── huatuo_encyclopedia_qa-filtered_flat.json      # Huatuo encyclopedia Q&A (filtered)
│   ├── huatuo_knowledge_graph_qa-filtered_flat.json   # Huatuo knowledge graph Q&A (filtered)
│   ├── Huatuo26M-Lite-filtered_flat.json      # Huatuo-26M-Lite Q&A (filtered)
│   ├── medical_o1_sft_Chinese-filtered.json   # Chinese medical o1-style SFT data (filtered)
│   ├── train_CMExam_single_sft.json           # CMExam single-choice training set (SFT format)
│   ├── multimedqa_sft.json                    # MultiMedQA (SFT format)
│   ├── medbullets_sft.json                    # MedBullets Q&A (SFT format)
│   ├── CMB_multiple_sft.json                  # CMB multiple-choice training set (SFT format)
│   ├── train_CMB_sin_sft.json                 # CMB single-choice training set (SFT format)
│   ├── CMExam_multiple_sft.json               # CMExam multiple-choice training set (SFT format)
│   ├── medqa_train.json                       # MedQA (USMLE) training set
│   └── medmcqa_train.json                     # MedMCQA training set
├── benchmark/
│   ├── checked_converted_medmcqa_test.json    # MedMCQA benchmark question set (270 questions)
│   ├── checked_converted_medqa_test.json      # MedQA (USMLE) benchmark question set (138 questions)
│   ├── checked_merged_CMExam_test.json        # CMExam benchmark question set
│   ├── checked_merged_test_CMB.json           # CMB benchmark question set
│   ├── evaluate_benchmark.py                  # Benchmark model inference and 95% CI computation
│   └── plot_benchmark.py                      # Benchmark accuracy bar chart (Fig. 7) from pre-computed CSV
├── expert_pilot/
│   └── plot_expert_evaluation.py              # Expert pilot summaries, Friedman/Wilcoxon tests, and grouped bar chart (Fig. 8)
├── clinical_trial/
│   ├── baseline_characteristics.py            # Baseline demographics tables (Tables 1 and 2)
│   ├── weight_loss_analysis.py                # Weight-loss outcomes bar chart (Fig. 4)
│   └── glycemic_control_analysis.py           # Fasting glucose outcomes bar chart (Fig. 5)
├── questionnaire/
│   └── participant_reported.py                # Participant-reported outcomes radar chart (Fig. 6)
└── Subgroup Forest Plot/
    ├── weight-loss subgroup forest plot.py    # Subgroup forest plot for weight-loss cohort
    └── glycemic control subgroup forest plot.py  # Subgroup forest plot for glycemic-control cohort
```

## System Requirements

- **Python**: 3.9 or later
- **Operating system**: Tested on Ubuntu 22.04; compatible with macOS and Windows
- **Hardware**: Local model inference (benchmark evaluation) requires a CUDA-capable GPU with at least 16 GB VRAM for 14B-parameter models. All statistical analysis and plotting scripts run on CPU.
- **Dependencies**: All required packages are listed in `requirements.txt`. Key packages include `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `torch`, and `transformers`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ling112211/Exercise-Prescription-System.git
   cd Exercise-Prescription-System
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Installation should take less than 10 minutes on a standard machine with a stable internet connection.

## Data Status: Real vs. Example

> **Important**: Not all scripts in this repository can be run with real data out of the box. Please read this section carefully before running any scripts.

| Script | Output | Data in this repo |
|--------|--------|-------------------|
| `benchmark/plot_benchmark.py` | Fig. 7 | `data/benchmark_results/benchmark_accuracy.csv` (**real**) |
| `expert_pilot/plot_expert_evaluation.py` | Fig. 8 | `data/expert_pilot/*.csv` (**real**) |
| `clinical_trial/baseline_characteristics.py` | Tables 1 & 2 | `data/example/` (**example only**) |
| `clinical_trial/weight_loss_analysis.py` | Fig. 4 | `data/example/` (**example only**) |
| `clinical_trial/glycemic_control_analysis.py` | Fig. 5 | `data/example/` (**example only**) |
| `questionnaire/participant_reported.py` | Fig. 6 | `data/example/` (**example only**) |
| `Subgroup Forest Plot/*.py` | Extended Data Figs. 3-4 | `data/example/` (**example only**) |

The example data under `data/example/` are anonymised synthetic files provided solely to verify that the code runs without errors. They do **not** reproduce the numerical results or figures reported in the paper. To obtain the real clinical trial data (weight-loss RCT, glycemic-control RCT, and questionnaire), please contact the corresponding author (see [Data Availability](#data-availability)).

## How to Reproduce the Results

All scripts are run from the repository root. Data files must first be obtained (see [Data Availability](#data-availability)).

### Benchmark Evaluation (Fig. 7)

Fig. 7 is produced in two steps: (1) running model inference to compute accuracy scores and export the plotting CSV, and (2) plotting the results from the saved CSV. If you only want to reproduce the figure from the pre-computed results, skip to Step 2.

**Step 1 — Run model inference** (requires GPU; skip if using pre-computed CSV)

Evaluates EPS variants and baseline models across four medical benchmarks. Local models are loaded via HuggingFace; flagship model APIs require environment variables for API keys.

```bash
# Set API keys for proprietary models (optional, skip if only evaluating local models)
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."

# Run benchmark evaluation (10 runs per model per benchmark by default)
python benchmark/evaluate_benchmark.py --n_runs 10 --save_details
```

Before running, update the model path placeholders in `benchmark/evaluate_benchmark.py` (`LOCAL_MODEL_SPECS`) to point to your local model checkpoints or HuggingFace model IDs.

**Step 2 — Plot the bar chart from pre-computed CSV**

A pre-computed accuracy CSV is provided at `data/benchmark_results/benchmark_accuracy.csv`. Use `plot_benchmark.py` to generate Fig. 7 directly without re-running inference:

```bash
python benchmark/plot_benchmark.py \
    --input  data/benchmark_results/benchmark_accuracy.csv \
    --outdir outputs/benchmark
```

Both `--input` and `--outdir` have the above defaults and may be omitted when running from the repository root.

### Expert Pilot Evaluation (Fig. 8)

Computes descriptive mean scores with two-sided 95% t-based confidence intervals, runs Friedman omnibus tests with Holm adjustment across the seven dimensions, runs paired Wilcoxon signed-rank tests for the three prespecified pairwise comparisons, and generates the grouped bar chart for Fig. 8.

```bash
python expert_pilot/plot_expert_evaluation.py \
    --input-dir data/expert_pilot \
    --pair-key rater_id \
    --outdir outputs/expert_pilot
```

Expected input files under `data/expert_pilot/`: `base_model.csv`, `eps_without_d2.csv`, `eps.csv`. Each CSV contains a shared rater identifier column (`rater_id` in the bundled data) and seven question columns (Q1–Q7) with A/B/C/D grades or numeric 0–3 scores from 25 health managers.

The script writes the following files to the output directory:
- `<prefix>_means_ci.csv` — descriptive means, SDs, and 95% confidence intervals.
- `<prefix>_aligned_scores.csv` — the paired analysis table after aligning raters across the three files.
- `<prefix>_friedman_tests.csv` — omnibus Friedman test results with Holm adjustment across the seven dimensions.
- `<prefix>_wilcoxon_pairwise_tests.csv` — paired Wilcoxon results with raw P values plus two Holm-adjusted columns: `p_holm_3pairs_within_dimension` (the manuscript reporting column) and `p_holm_7dims_within_comparison` (exported for transparency).
- `<prefix>_bar_mean_ci.pdf` and `<prefix>_bar_mean_ci.png` — the grouped bar chart used for Fig. 8.

### Baseline Characteristics (Tables 1 and 2)

Generates the demographic comparison tables for both trial cohorts.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Tables 1 and 2 in the paper. Replace the paths with your real data files once access has been granted.

```bash
python clinical_trial/baseline_characteristics.py \
    --weight_human data/example/weight_loss/human_arm.xlsx \
    --weight_eps   data/example/weight_loss/eps_arm.xlsx \
    --gly_human    data/example/glycemic/human_arm.xlsx \
    --gly_eps      data/example/glycemic/eps_arm.xlsx \
    --out_dir      outputs/clinical_trial
```

### Weight-Loss Outcomes (Fig. 4)

Computes Welch *t*-test statistics and Clopper–Pearson confidence intervals, then generates the three-panel bar chart.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Fig. 4 in the paper. Replace the paths with your real data files once access has been granted.

```bash
python clinical_trial/weight_loss_analysis.py \
    --weight_human data/example/weight_loss/human_arm.xlsx \
    --weight_eps   data/example/weight_loss/eps_arm.xlsx \
    --out_dir      outputs/clinical_trial
```

### Glycemic-Control Outcomes (Fig. 5)

Computes fasting glucose reduction statistics and generates the two-panel bar chart.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Fig. 5 in the paper. Replace the paths with your real data files once access has been granted.

```bash
python clinical_trial/glycemic_control_analysis.py \
    --gly_human data/example/glycemic/human_arm.xlsx \
    --gly_eps   data/example/glycemic/eps_arm.xlsx \
    --out_dir   outputs/clinical_trial
```

### Participant-Reported Outcomes (Fig. 6)

Applies quality-control filters (Q1 screening, completion-time filter, straight-lining detection) and generates the radar chart with 95% confidence intervals.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Fig. 6 in the paper. Replace the paths with your real data files once access has been granted.

```bash
python questionnaire/participant_reported.py \
    --human-xlsx data/example/questionnaire/human_responses.xlsx \
    --eps-xlsx   data/example/questionnaire/eps_responses.xlsx \
    --outdir     outputs/questionnaire
```

### Subgroup Forest Plots (Extended Data Figs. 3 and 4)

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Extended Data Figs. 3 and 4 in the paper. Replace the paths with your real data files once access has been granted.

```bash
# Weight-loss subgroup analysis
python "Subgroup Forest Plot/weight-loss subgroup forest plot.py" \
    --eps   data/example/weight_loss/eps_arm.xlsx \
    --human data/example/weight_loss/human_arm.xlsx \
    --out-prefix outputs/subgroup/weightloss_subgroup

# Glycemic-control subgroup analysis
python "Subgroup Forest Plot/glycemic control subgroup forest plot.py" \
    --eps      data/example/glycemic/eps_arm.xlsx \
    --human    data/example/glycemic/human_arm.xlsx \
    --col_bmi  BMI \
    --out_table outputs/subgroup/glycemic_subgroup.xlsx \
    --out_png   outputs/subgroup/glycemic_subgroup.png \
    --out_pdf   outputs/subgroup/glycemic_subgroup.pdf
```

## Data Availability

This repository includes two categories of data:

**Fully available (real data, reproduces paper results):**
- `D1/` — public exercise and weight-management corpus (86,900 entries across 18 datasets) used for supervised fine-tuning of EPS. Assembled from publicly available medical instruction datasets and filtered using a bilingual keyword lexicon to retain exercise- and weight-management–relevant instances (see Section 4.3.1 and Supplementary Table 1 for full keyword list and matching rules). Extended Data Table 1 in the paper summarises D1 composition by target application, language, and number of examples.
- `benchmark/checked_converted_medmcqa_test.json` — MedMCQA benchmark question set (270 questions) used for model evaluation.
- `benchmark/checked_converted_medqa_test.json` — MedQA (USMLE) benchmark question set (138 questions) used for model evaluation.
- `benchmark/checked_merged_CMExam_test.json` — CMExam benchmark question set used for model evaluation.
- `benchmark/checked_merged_test_CMB.json` — CMB benchmark question set used for model evaluation.
- `data/benchmark_results/benchmark_accuracy.csv` — pre-computed benchmark accuracy scores used to generate Fig. 7.
- `data/expert_pilot/` — expert ratings from the 25-person pilot study used to generate Fig. 8.

**Example data only (does not reproduce paper results):**
- `data/example/` — anonymised synthetic datasets provided solely to verify that the analysis and plotting scripts run without errors. These files have the same format as the real data but contain different values. Outputs produced with these files will **not** match the tables and figures reported in the paper.

The real clinical trial data (weight-loss RCT: 1,444 participants; glycemic-control RCT: 40 participants; participant questionnaire) are available under controlled access due to patient privacy regulations. Researchers who wish to access the de-identified participant data for academic purposes may contact the corresponding author. Please see `data/README_data.md` for a full description of each dataset and the required file format.

## Model Availability

EPS models are fine-tuned versions of open-source base models (DeepSeek-R1-8B, Qwen3-8B, DeepSeek-R1-14B, Qwen3-14B) using a two-stage alignment framework: supervised fine-tuning on domain-specific data (D1) followed by Kahneman–Tversky Optimization on expert-preference data (D2). Model checkpoints are not publicly released due to the use of proprietary training data. Qualified researchers may request access by contacting the corresponding author.

Base models are available from their official repositories:
- **Qwen3**: [github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) (Apache-2.0 License)
- **DeepSeek-R1**: [github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) (MIT License)

## Ethics and Trial Registration

The study protocol was approved by the Ethics Review Committees of City University of Hong Kong, Harbin Institute of Technology, and Ping An Health and Technology Co., Ltd. The trial was registered at the [Chinese Clinical Trial Registry (ChiCTR2600118939)](https://www.chictr.org.cn/hvshowprojectEN.html?id=295346&v=1.0). All participants provided written informed consent prior to enrollment.

## How to Cite

If you use this code in your research, please cite our paper (citation details will be updated upon publication):

```bibtex
@article{,
  author  = {},
  title   = {An LLM-Based Exercise Prescription System for Digital Chronic Disease Management},
  journal = {},
  year    = {2025},
  doi     = {}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions regarding the code, data access, or model access, please contact:

- **Guangxin Jiang** (corresponding author): gxjiang@hit.edu.cn
- **Chenxi Li**: ling112358@gmail.com
- **Siyang Gao**: siyangao@cityu.edu.hk
