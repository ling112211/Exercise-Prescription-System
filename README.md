# Exercise-Prescription-System

Code and data to reproduce the analyses, tables, and figures reported in:



<small>[Chinese Clinical Trial Registry identifier](https://www.chictr.org.cn/showproj.html?proj=310458)</small>
---

## Overview

This repository contains the source code for reproducing the statistical analyses, benchmark evaluations, and visualizations presented in the paper. The Exercise Prescription System (EPS) is an expert-aligned large language model (LLM)–based system that produces safe, personalized, and scalable exercise feedback for digital chronic disease management. EPS was evaluated through domain-specific medical benchmarks, an expert pilot study with 25 professional health managers, and a single-blind randomized controlled trial involving 1,444 weight-loss participants and 40 glycemic-control participants.

## Repository structure

```
Exercise-Prescription-System/
├── README.md
├── LICENSE
├── requirements.txt
├── benchmark/
│   └── evaluate_benchmark.py        # Accuracy and 95% CI computation for EPS and flagship LLMs
├── expert_pilot/
│   └── plot_expert_evaluation.py    # Grouped bar chart for Fig. 3
├── clinical_trial/
│   ├── baseline_characteristics.py  # Demographic comparisons (Tables 1 and 2)
│   ├── weight_loss_analysis.py      # Weight-loss outcomes (Fig. 4)
│   └── glycemic_control_analysis.py # Fasting glucose outcomes (Fig. 5)
├── questionnaire/
│   └── participant_reported.py      # Radar plot for Fig. 6
└── data/
    └── README_data.md               # Data dictionary and access instructions
```

## System requirements

### Software dependencies

- Python 3.9 or later
- Operating system: tested on Ubuntu 22.04 (Linux); expected to be compatible with macOS and Windows

### Python packages

All required packages are listed in `requirements.txt`. Key dependencies include:

- `numpy>=1.25.0`
- `scipy>=1.11.0`
- `pandas>=2.0.0`
- `matplotlib>=3.7.0`
- `statsmodels>=0.14.0`

### Installation

```bash
git clone https://github.com/XXXX/Exercise-Prescription-System.git
cd Exercise-Prescription-System
pip install -r requirements.txt
```

Typical installation time on a standard desktop is under 5 minutes.

## Data availability

### Public exercise and weight-management corpus (D1)

The public training corpus D1 (86,900 question–answer entries) was assembled by filtering publicly available medical instruction datasets with a bilingual keyword lexicon. The names and sources of all constituent datasets are listed in Extended Data Table B1 of the paper. The keyword lexicon used for filtering is provided in Supplementary Table 1. Each constituent dataset should be obtained from its original source and used under its original license (see Extended Data Table B1 for the source list).

### Expert-curated exercise prescription dataset (D2)

The expert-curated dataset D2 (1,156 entries) contains population-level exercise guidance, user-personalized exercise suggestions, expert exercise reviews, and an exercise type database. Because D2 was curated by professional health managers and contains proprietary coaching content, it is not publicly released. Requests for access to de-identified D2 data for academic research purposes can be submitted via email to G.J. (gxjiang@hit.edu.cn) with a research proposal and justification for data use. All requests will be reviewed by the Ethics Review Committee of City University of Hong Kong. Review of proposals may take up to 2 months, and approved requests will require execution of a data access agreement.

### Filtered medical benchmarks

We constructed domain-specific test subsets from four established medical benchmarks by applying the bilingual keyword lexicon (Supplementary Table 1). The names and original sources of the benchmark datasets are listed in Extended Data Table D. 

### Clinical trial data

Individual-level clinical trial data supporting the findings of this study are available within the paper and its Supplementary Information. Source data for Tables 1–2 and Figs. 1–6 are provided via this repository. Raw participant data are not publicly available due to privacy restrictions, in accordance with the ethical approval for this study. Anonymized, individual-level data underlying the trial results can be requested by qualified researchers for academic use. Requests should include a research proposal, statistical analysis plan, and justification for data use, and can be submitted via email to G.J. (gxjiang@hit.edu.cn). All requests will be reviewed by the Ethics Review Committee of City University of Hong Kong and other participating centers. Review of proposals may take up to 2 months, and approved requests will be granted access after execution of a data access agreement.

## Code description

### Benchmark evaluation (`benchmark/`)

`evaluate_benchmark.py` reproduces the benchmark results reported in Table 3. The script computes accuracy (%) and 95% confidence intervals for both EPS models (DeepSeek-R1-8B, Qwen3-8B, DeepSeek-R1-14B, Qwen3-14B) and mainstream LLMs (ChatGPT-5, DeepSeek-R1, Gemini 2.5 Flash, Grok 4 Fast) across four filtered medical benchmarks (CMB, CMExam, MedMCQA, MedQA). Inference was repeated 10 times with different random seeds; the script reports mean accuracy across runs.

### Expert pilot evaluation (`expert_pilot/`)

`plot_expert_evaluation.py` generates the grouped bar chart (Fig. 3), displaying mean scores (0–3 scale) across seven evaluation dimensions (consensus, correctness, completeness, unbiasedness, clarity, empathy, and actionability) for the Base model, EPS without D2, and EPS. Error bars denote 95% confidence intervals. Twenty-five professional health managers independently rated exercise-feedback outputs; pairwise comparisons use two-sided tests with Holm step-down adjustment.

### Clinical trial analyses (`clinical_trial/`)

- **`baseline_characteristics.py`**: Reproduces Tables 1 and 2. Compares baseline demographics (age, sex, BMI, baseline weight or fasting glucose) between the Human arm and the EPS–human arm for both the weight-loss cohort (*n* = 1,444) and the glycemic-control cohort (*n* = 40). Continuous variables are compared using two-sided Welch's *t*-tests; categorical variables are compared using Pearson's χ² tests or Fisher's exact tests.

- **`weight_loss_analysis.py`**: Reproduces Fig. 4. Computes between-arm differences in mean weight loss (kg), percent weight loss, and the proportion achieving ≥2% weight loss. *P* values are from Welch's *t*-tests (continuous outcomes) and a two-proportion *z*-test (binary outcome). The script also performs the prespecified covariate-adjusted sensitivity analysis using ordinary least squares regression.

- **`glycemic_control_analysis.py`**: Reproduces Fig. 5. Computes between-arm differences in fasting glucose reduction (mmol/L) and fasting glucose reduction ratio (%). In addition to Welch's *t*-tests, the script performs the bootstrap sensitivity analysis (*B* = 10,000 resamples, percentile intervals, fixed random seed) to assess robustness.

### Participant-reported outcomes (`questionnaire/`)

`participant_reported.py` generates the radar plot (Fig. 6), summarizing mean scores across 14 questionnaire dimensions (items 2–15) for the Human (*n* = 344 valid responses) and EPS–human (*n* = 336 valid responses) arms. Each item is scored on a 1–7 Likert scale. Between-arm comparisons use two-sided Welch's *t*-tests with Holm step-down adjustment for multiple testing.

## Reproducing the main results

To reproduce the main analyses and figures reported in the paper:

```bash
# Benchmark evaluation (Table 3)
python benchmark/evaluate_benchmark.py

# Expert pilot evaluation (Fig. 3)
python expert_pilot/plot_expert_evaluation.py

# Baseline characteristics (Tables 1 and 2)
python clinical_trial/baseline_characteristics.py

# Weight-loss outcomes (Fig. 4)
python clinical_trial/weight_loss_analysis.py

# Glycemic-control outcomes (Fig. 5)
python clinical_trial/glycemic_control_analysis.py

# Participant-reported outcomes (Fig. 6)
python questionnaire/participant_reported.py
```

Expected run time for the complete statistical analysis pipeline is under 10 minutes on a standard desktop computer. Benchmark inference time depends on hardware and model size.

## EPS model availability

The EPS model used in the pilot study and the randomized controlled trial was a fine-tuned Qwen3-14B model (EPS-Qwen3-14B). The model checkpoint and inference configuration were fixed before study initiation. EPS is not publicly released at this time because it incorporates proprietary expert-curated training data and is subject to ongoing deployment considerations. To support academic validation and collaborative research, the EPS model can be made available to qualified researchers upon a formal request to G.J. (gxjiang@hit.edu.cn), subject to a data-sharing agreement, ethical approvals, and a commitment to appropriate safety protocols. Base models are available from their official repositories: [Qwen3](https://github.com/QwenLM/Qwen3) (Apache-2.0 license) and [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) (MIT license).

## Ethics and trial registration

The study protocol covered both the pilot study and the randomized controlled trial, and was reviewed and approved by the Ethics Review Committee of City University of Hong Kong and other participating centers. The trial was prospectively registered at the Chinese Clinical Trial Registry (identifier [ChiCTR2600118939](https://www.chictr.org.cn/showproj.html?proj=XXXXX)). Written informed consent was obtained from all participants before enrollment. All procedures were conducted in accordance with the Declaration of Helsinki and the International Council for Harmonisation Good Clinical Practice (ICH-GCP) guidelines.

## Citation

If you use this code or data in your work, please cite:

```bibtex

```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions regarding the code, data access, or the EPS model, please contact:

- **Guangxin Jiang** (corresponding author): gxjiang@hit.edu.cn
- **Chenxi Li**: ling112358@gmail.com
