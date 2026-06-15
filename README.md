# Policy Negotiation in Data Trading

This repository contains the source code, representative outputs, and reproducibility materials for the paper:

**Usage Policies Negotiation in Data Trading with Time-Discounting Valuations: A Multi-Stage Stackelberg Game Approach**

The code implements a three-stage Stackelberg policy negotiation model for time-sensitive data trading, together with comparison experiments, functional-form robustness experiments, and repeated statistical robustness experiments.

## Repository Structure

```text
src/
  data_trading_game.py              Main Stackelberg model and core experiment utilities
  comparison_experiment.py          Revised comparison experiment for Proposed/SRC/BGM/APM
  robustness_quality_decay.py       Functional-form robustness experiments
  robustness_utils.py               Dataset profiling, repeated trials, CIs, significance tests
  run_robustness_experiments.py     CLI entry point for statistical robustness experiments

data/
  Small processed CSV files used by the scripts

results/
  Representative CSV and audit outputs

figures/
  Representative generated figures in PDF format
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Quick Start

Run the comparison experiment:

```bash
python src/comparison_experiment.py
```

Run functional-form robustness experiments:

```bash
python src/robustness_quality_decay.py calibrate-exp
python src/robustness_quality_decay.py power-calibrate
python src/robustness_quality_decay.py delivery-power-calibrate
python src/robustness_quality_decay.py delivery-piecewise-calibrate
python src/robustness_quality_decay.py risk-cost-calibrate
```

Run a statistical robustness experiment:

```bash
python src/run_robustness_experiments.py \
  --datasets amazon_employee \
  --num-runs 10 \
  --overlap-levels medium \
  --base-seed 2026 \
  --output-dir results/robustness
```

The full repeated experiment used for the paper can be run as:

```bash
python src/run_robustness_experiments.py \
  --datasets amazon_employee uci_amazon_access incident_event_log \
  --num-runs 30 \
  --overlap-levels low medium high \
  --base-seed 2026 \
  --output-dir results/robustness
```

See `DATA.md` for dataset download and placement instructions.

## Reproducibility Outputs

Representative outputs are included under:

```text
results/comparison_revised/
results/robustness_quality_decay/
results/robustness/
figures/
```

Only representative PDF figures are included. In particular, `figures/robustness/` keeps the representative statistical-robustness figure used in the manuscript; the full set of dataset-by-overlap figures can be regenerated from the scripts and is not stored by default to avoid unnecessary files.
