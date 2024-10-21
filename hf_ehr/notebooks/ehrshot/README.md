# EHRSHOT Results

This folder contains plots and tables containing EHRSHOT results of models.

### Motivation
Evaluate each model on 14 (non-Chexpert) EHRSHOT tasks.

### Usage
Each notebook can be run in parallel. They do the following:
* `ehrshot_plots.ipynb` -- Plot AUROC performance for each (model, context length) across (a) each individual tasks; (b) all tasks averaged together
* `ehrshot_results.ipynb` -- Calculate AUROC of each model, 1k bootstrap of AUROC diff v. baseline model, win rate v. baseline model
* `ehrshot_results_stratification.ipynb` -- Calculate Brier scores of each model across quartiles for each EHR-specific property
* `ehrshot_stats.ipynb` -- EDA of EHRSHOT dataset, e.g. token counts, etc.

### Paper
See **Figure 1b** and **Appendix Figure 8** and **Appendix Tables XX-XX**
