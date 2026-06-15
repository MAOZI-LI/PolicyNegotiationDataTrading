# Reproducibility Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Comparison Experiment

```bash
python src/comparison_experiment.py
```

Outputs:

```text
results/comparison_revised/comparison_single_run.csv
results/comparison_revised/comparison_audit.txt
figures/comparison_revised/compare_revised.png
figures/comparison_revised/compare_revised.pdf
```

## 3. Functional-Form Robustness Experiments

```bash
python src/robustness_quality_decay.py calibrate-exp
python src/robustness_quality_decay.py power-calibrate
python src/robustness_quality_decay.py delivery-power-calibrate
python src/robustness_quality_decay.py delivery-piecewise-calibrate
python src/robustness_quality_decay.py risk-cost-calibrate
```

Outputs are written to:

```text
results/robustness_quality_decay/
figures/robustness_quality_decay/
```

## 4. Statistical Robustness Experiments

Quick example:

```bash
python src/run_robustness_experiments.py \
  --datasets amazon_employee \
  --num-runs 10 \
  --overlap-levels medium \
  --base-seed 2026 \
  --output-dir results/robustness
```

Full setting:

```bash
python src/run_robustness_experiments.py \
  --datasets amazon_employee uci_amazon_access incident_event_log \
  --num-runs 30 \
  --overlap-levels low medium high \
  --base-seed 2026 \
  --output-dir results/robustness
```

Outputs:

```text
results/robustness/dataset_profiles.csv
results/robustness/robustness_raw_results.csv
results/robustness/robustness_summary.csv
results/robustness/significance_tests.csv
results/robustness/figures/
```

