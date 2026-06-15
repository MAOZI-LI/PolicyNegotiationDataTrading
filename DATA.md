# Data

This repository includes small processed CSV files needed by the main scripts.

Large raw datasets are not included. To run all statistical robustness experiments, download the following datasets from their official sources and place them under a local `dataset/` directory:

```text
dataset/amazon+access+samples.zip
dataset/incident+management+process+enriched+event+log.zip
```

The statistical robustness script expects:

- `amazon_employee`: `data/over-datasets-kaggle-log-clean.csv`
- `uci_amazon_access`: `dataset/amazon+access+samples.zip`, using inner file `amzn-anon-access-samples-history-2.0.csv`
- `incident_event_log`: `dataset/incident+management+process+enriched+event+log.zip`, using inner file `incident_event_log.csv`

The UCI Amazon Access experiment uses the smaller history CSV inside the downloaded zip rather than the multi-GB main file, so repeated experiments remain computationally manageable.

