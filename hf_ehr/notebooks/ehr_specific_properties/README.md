# EHR-Specific Property EHRSHOT Stratification Experiments

### Motivation

Our goal is to stratify EHRSHOT labels based on several EHR-specific metrics into quartiles, then recalculate their EHRSHOT Brier scores and see how context length / model impacts performance across quartiles.

The metrics we use to quartile the EHRSHOT labels are:
* **Repetitiveness:** `rr_1` -- repetition rate of 1-grams
* **Irregularity:** `std` -- std in inter-event times
* **Timeline Lengths:** `n_tokens` -- number of tokens that model can ingest for prediction

### Usage

```bash
# First, calculate each stratification metric across all EHRSHOT labels, i.e. unique (patient, label time) tuples
bash calc_stratification_metrics.sh

# Second, bucket EHRSHOT labels by each stratification metric
bash bucket_by_metrics.sh

# Third, calculate Brier scores + bootstraps
bash calc_brier_scores.sh
```

### Paper
See **Section 4 (Results)** in the paper.
