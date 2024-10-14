import pandas as pd
import scipy.stats as stats

# Read the master CSV file
df = pd.read_csv('/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/master_metrics_1000.csv')

# Filter the DataFrame to include only 'time_std' and 'rr_1' stratification metrics
df = df[df['strat_metric'].isin(['time_std', 'rr_1'])]

# Group by model, strat_metric, quartile, and bootstrap to calculate mean Brier score across tasks
grouped = df.groupby(['model', 'strat_metric', 'quartile', 'bootstrap'])['brier_score'].mean().reset_index()

# Get the list of models
models = grouped['model'].unique()
if len(models) < 2:
    print("Not enough models to compare.")
    exit()

print(models)
# If you have more than two models, specify which ones to compare
# You can modify these variables to choose the models you want to compare
model1 = models[2]
model2 = models[3]

# Get the list of stratification metrics and quartiles
strat_metrics = ['time_std', 'rr_1']  # Only these two stratification metrics
quartiles = grouped['quartile'].unique()

results = []

for strat_metric in strat_metrics:
    for quartile in quartiles:
        # Get data for this stratification metric and quartile
        data = grouped[(grouped['strat_metric'] == strat_metric) & (grouped['quartile'] == quartile)]
        
        # Ensure both models are present in the data
        models_present = data['model'].unique()
        if model1 not in models_present or model2 not in models_present:
            continue  # Skip if both models are not present
        
        # Get Brier scores for each model
        brier_scores_model1 = data[data['model'] == model1]['brier_score']
        brier_scores_model2 = data[data['model'] == model2]['brier_score']
        
        # Ensure there are enough data points
        if len(brier_scores_model1) < 2 or len(brier_scores_model2) < 2:
            continue  # Skip if not enough data to perform t-test
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(brier_scores_model1, brier_scores_model2)
        
        # Compute mean Brier scores
        mean_brier_score_model1 = brier_scores_model1.mean()
        mean_brier_score_model2 = brier_scores_model2.mean()
        
        # Compute mean difference
        mean_diff = mean_brier_score_model1 - mean_brier_score_model2
        
        # Store results
        results.append({
            'strat_metric': strat_metric,
            'quartile': quartile,
            'model1': model1,
            'model2': model2,
            'mean_brier_score_model1': mean_brier_score_model1,
            'mean_brier_score_model2': mean_brier_score_model2,
            'mean_diff': mean_diff,
            't_stat': t_stat,
            'p_value': p_value,
            'n_model1': len(brier_scores_model1),
            'n_model2': len(brier_scores_model2)
        })

# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('model_comparison_results_hyena_1000.csv', index=False)

# Optionally, print the results
print(results_df)
