## Closer to Home: A Structural Estimate-then-Optimize Approach to Improve Access to Healthcare Services

This repository contains all the code used in the numerical experiments in the paper: 
> Bravo, F., Hu, J., & Long, E. F. (2025). Predicting long-term opioid use risk via interpretable machine learning



### Overview
This repository provides all necessary scripts from the paper. The code includes steps for data processing/engineering, numerical experiments, and visualization.

Workflow
- Data processing: 
- Data Preparation: Download, preprocess and structure the required datasets.
- Demand Estimation: Run demand models for analysis.
- Optimization & Evaluation: Solve location optimization problems and assess outcomes.
- Summary results and visualization: Generate summary tables and figures for the paper.


### Software Requirements
RiskSLIM requires Python 3.5+ and CPLEX 12.6+. For download and installation instructions, click here. For more detailed guidance on installing and running RiskSLIM, please refer to the GitHub repository by [Ustun and Rudin (2019)](https://github.com/ustunb/risk-slim). In addition, [Wang, Han, Patel, and Rudin (2023)](https://github.com/BeanHam/2019-interpretable-machine-learning) provide an excellent example of using RiskSLIM alongside other benchmark ML models, which we have partially adapted.



### Folder and File Structure
The repository is organized into the following main directories and files:

```
├── README.md
├── requirements.txt
├── riskslim/                # Migrated from the original RiskSLIM repo by Ustun & Rudin (2019)
├── setup.py                 # Environment setup (migrated from original repo)
│
├── risk_stumps.py           # Create stumps as inputs
├── risk_main.py             # RiskSLIM main execution
├── risk_test.py             # RiskSLIM test
├── risk_train.py            # RiskSLIM train
│
├── sh/                      # Shell scripts for running experiments
│   ├── baseline.sh
│   ├── county.sh
│   ├── create_stumps.sh
│   ├── data_process.sh
│   ├── explore.sh
│   └── riskslim.sh
├── src/                     # Scripts for figures and data processing
│   ├── barplot.R
│   ├── convert_R.py
│   ├── plot_baseline.py
│   ├── plot_roc_calibration_proportion.R
│   ├── process/             # Data cleaning, filtering, and feature engineering
│   │   ├── step1_filter_split_data.py
│   │   ├── step23_identify_alert_lt_multiple.py
│   │   ├── step23_identify_alert_lt_single.py
│   │   ├── step4_compute_features.py
│   │   ├── step5_compute_features_patient_prescriber.py
│   │   ├── step6_to_input.py
│   │   ├── step7_final_check.py
│   │   └── testdiff.py
│   └── utils/               # Generate summary tables for the paper
│       ├── county_summary.py
│       ├── export_baseline.py
│       ├── merge_tables.py
│       ├── raw_data_stats.py
│       ├── summary_stats_CURES.py
│       ├── summary_stats_prescriber_dispenser.py
│       └── summary_stats.py
└── utils/                   # Helper functions for benchmark model testing (adapted from Wang et al., 2023)
    ├── baseline_functions.py
    ├── fairness_functions.py
    ├── model_selection.py
    ├── RiskSLIM.py
    └── stumps.py
```
**Please ensure code is placed correctly before running scripts.**

This repository integrates and extends the original RiskSLIM implementation by Ustun and Rudin (2019) with additional preprocessing, training, and evaluation tools. The top-level scripts (risk_main.py, risk_train.py, risk_test.py, and risk_stumps.py) handle model execution, training, testing, and stump construction. The sh/ folder provides shell scripts to streamline experiment workflows. The src/ folder contains scripts for figure generation (in both R and Python) and a structured process/ pipeline for data cleaning, filtering, and feature engineering, while src/utils/ generates summary tables used in the paper. Finally, the utils/ directory includes helper functions for benchmarking and fairness evaluation, adapted from Wang, Han, Patel, and Rudin (2023).


