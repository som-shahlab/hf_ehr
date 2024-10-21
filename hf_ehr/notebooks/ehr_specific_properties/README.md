# EHR-specific property stratification experiments

We stratify patients based on these metrics, then recalculate their EHRSHOT Brier scores across all models:
* **Repetitiveness:** `rr_1` -- repetition rate of 1-grams
* **Irregularity:** `std` -- std in inter-event times
* **Timeline Lengths:** `n_tokens` -- number of tokens that model can ingest for prediction

```bash
# First, calculate each stratification metric across all EHRSHOT labels, i.e. unique (patient, label time) tuples
bash calc_stratification_metrics.sh

# Second, bucket EHRSHOT labels by each stratification metric
bash bucket_by_metrics.sh

# Third, calculate Brier scores + bootstraps
bash bootstrap.sh
```
