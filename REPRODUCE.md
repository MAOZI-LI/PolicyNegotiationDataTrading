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
figures/comparison_revised/compare_revised.pdf
```

## 3. Functional-Form Robustness Experiments

```bash
python src/functional_form_robustness.py calibrate-exp
python src/functional_form_robustness.py power-calibrate
python src/functional_form_robustness.py delivery-power-calibrate
python src/functional_form_robustness.py delivery-piecewise-calibrate
python src/functional_form_robustness.py risk-cost-calibrate
```

Outputs are written to:

```text
results/functional_form_robustness/
figures/functional_form_robustness/
```

## 4. Statistical Robustness Experiments

Quick example:

```bash
python src/run_robustness_experiments.py \
  --datasets amazon_employee \
  --num-runs 10 \
  --overlap-levels medium \
  --base-seed 2026 \
  --output-dir results/statistical_robustness
```

Full setting:

```bash
python src/run_robustness_experiments.py \
  --datasets amazon_employee uci_amazon_access incident_event_log \
  --num-runs 30 \
  --overlap-levels low medium high \
  --base-seed 2026 \
  --output-dir results/statistical_robustness
```

Outputs:

```text
results/statistical_robustness/dataset_profiles.csv
results/statistical_robustness/robustness_raw_results.csv
results/statistical_robustness/robustness_summary.csv
results/statistical_robustness/significance_tests.csv
results/statistical_robustness/figures/
```
