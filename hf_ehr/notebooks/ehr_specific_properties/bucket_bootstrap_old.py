import pandas as pd
from scipy import stats
import numpy as np
from tqdm import tqdm

np.random.seed(42)

# Define columns to load
use_columns = ['sub_task', 'metric_name', 'brier_score', 'quartile']

# Define data types to reduce memory usage
dtypes = {
    'sub_task': 'category',
    'metric_name': 'category',
    'brier_score': 'float32',
    'quartile': 'float32'
}

# Set up file paths
file_512 = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__per_patient__all_tasks.csv'
file_4096 = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__per_patient__all_tasks.csv'

# Initialize dictionaries to store sampled dataframes for each metric
df_4096_brier = {}
df_512_brier = {}

# Set sample size per metric
sample_size = 50000
bootstrap_iterations = 5000  # Number of bootstrap samples for confidence intervals

# List of metrics to focus on
focus_metrics = ['time_std', 'ttr']

# Helper function to keep a rolling sample per metric while reading chunks
def random_sample_chunk(existing_df, new_chunk, sample_size):
    combined_df = pd.concat([existing_df, new_chunk])
    if len(combined_df) > sample_size:
        combined_df = combined_df.sample(n=sample_size, random_state=42)  # Randomly sample to maintain 10k rows
    return combined_df

# Define the chunksize
chunksize = 10**6  # Load 1 million rows at a time

# Track which metrics have already reached the target sample size
metrics_complete_4096 = set()
metrics_complete_512 = set()

# Load and filter the 4096 context model data, stop when enough rows are collected
print("Processing and sampling 4096 context model data...")
for chunk in tqdm(pd.read_csv(file_4096, chunksize=chunksize, usecols=use_columns, dtype=dtypes), desc="4096 context"):
    # Filter by the metrics 'time_std' and 'ttr'
    filtered_chunk = chunk[chunk['metric_name'].isin(focus_metrics)]
    
    for metric in filtered_chunk['metric_name'].unique():
        if metric in metrics_complete_4096:
            continue  # Skip if enough data has been collected for this metric

        metric_data = filtered_chunk[filtered_chunk['metric_name'] == metric]
        
        # For each metric, sample data across all subtasks
        if metric not in df_4096_brier:
            df_4096_brier[metric] = metric_data
        else:
            df_4096_brier[metric] = random_sample_chunk(df_4096_brier[metric], metric_data, sample_size)

        # If we have enough rows for this metric, mark it as complete
        if len(df_4096_brier[metric]) >= sample_size:
            metrics_complete_4096.add(metric)

    # Stop loading if all metrics have enough data
    if len(metrics_complete_4096) == len(focus_metrics):
        break

# Load and filter the 512 context model data, stop when enough rows are collected
print("Processing and sampling 512 context model data...")
for chunk in tqdm(pd.read_csv(file_512, chunksize=chunksize, usecols=use_columns, dtype=dtypes), desc="512 context"):
    # Filter by the metrics 'time_std' and 'ttr'
    filtered_chunk = chunk[chunk['metric_name'].isin(focus_metrics)]
    
    for metric in filtered_chunk['metric_name'].unique():
        if metric in metrics_complete_512:
            continue  # Skip if enough data has been collected for this metric

        metric_data = filtered_chunk[filtered_chunk['metric_name'] == metric]
        
        # For each metric, sample data across all subtasks
        if metric not in df_512_brier:
            df_512_brier[metric] = metric_data
        else:
            df_512_brier[metric] = random_sample_chunk(df_512_brier[metric], metric_data, sample_size)

        # If we have enough rows for this metric, mark it as complete
        if len(df_512_brier[metric]) >= sample_size:
            metrics_complete_512.add(metric)

    # Stop loading if all metrics have enough data
    if len(metrics_complete_512) == len(focus_metrics):
        break

# Perform one-way bootstrap t-tests for each metric_name and quartile
results = {}
for metric in df_4096_brier:
    quartiles = df_4096_brier[metric]['quartile'].unique()
    results[metric] = {}
    
    for quartile in quartiles:
        # Filter data for the specific quartile and metric, combining all subtasks
        brier_4096 = df_4096_brier[metric][df_4096_brier[metric]['quartile'] == quartile]['brier_score']
        brier_512 = df_512_brier[metric][df_512_brier[metric]['quartile'] == quartile]['brier_score']
        
        if len(brier_4096) == 0 or len(brier_512) == 0:
            print(f"Skipping one-way t-test for metric {metric}, quartile {quartile} due to insufficient data.")
            continue
        
        # Bootstrap resampling
        bootstrap_t_stats = []
        for _ in range(bootstrap_iterations):
            brier_4096_resample = np.random.choice(brier_4096, size=len(brier_4096), replace=True)
            brier_512_resample = np.random.choice(brier_512, size=len(brier_512), replace=True)
            t_stat, _ = stats.ttest_ind(brier_4096_resample, brier_512_resample, equal_var=False)
            bootstrap_t_stats.append(t_stat)
        
        # Confidence intervals for t-statistic
        ci_lower = np.percentile(bootstrap_t_stats, 2.5)
        ci_upper = np.percentile(bootstrap_t_stats, 97.5)
        
        # Perform one-way t-test on the actual data
        t_stat, p_value = stats.ttest_ind(brier_4096, brier_512, equal_var=False)
        
        # Adjust p-value for one-way test (since we're testing if 4096 < 512)
        if t_stat < 0:  # Only consider when 4096 < 512
            p_value /= 2
        else:
            p_value = 1.0  # If t-statistic is positive, p-value is set to 1
        
        # Store results
        results[metric][quartile] = {'t_stat': t_stat, 'p_value': p_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

# Display the results
for metric, quartile_results in results.items():
    print(f'Results for metric {metric}:')
    for quartile, result in quartile_results.items():
        print(f'  Quartile {quartile}: t-statistic = {result["t_stat"]}, one-sided p-value = {result["p_value"]}, 95% CI = ({result["ci_lower"]}, {result["ci_upper"]})')
