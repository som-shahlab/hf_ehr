import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Directory containing the CSV files
csv_dir = "/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/"

# List of files to process
selected_files = [
    'metrics__clmbr__guo_icu.csv',
    'metrics__clmbr__guo_los.csv',
    'metrics__clmbr__guo_readmission.csv',
    'metrics__clmbr__lab_anemia.csv',
    'metrics__clmbr__lab_hyperkalemia.csv',
    'metrics__clmbr__lab_hypoglycemia.csv',
    'metrics__clmbr__lab_hyponatremia.csv',
    'metrics__clmbr__lab_thrombocytopenia.csv',
    'metrics__clmbr__new_acutemi.csv',
    'metrics__clmbr__new_celiac.csv',
    'metrics__clmbr__new_hyperlipidemia.csv',
    'metrics__clmbr__new_hypertension.csv',
    'metrics__clmbr__new_lupus.csv',
    'metrics__clmbr__new_pancan.csv'
]

# Metrics to process
metrics = ['time_std', 'rr_1', 'n_events', 'n_unique_events']

# Initialize a list to store results for each file
file_results = []

# Process each file
for file in selected_files:
    file_path = os.path.join(csv_dir, file)
    if os.path.exists(file_path):
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Process each metric
        for metric in metrics:
            metric_data = data[data['strat_metric'] == metric]
            metric_mean = metric_data.groupby('quartile')['brier_score'].mean()
            
            # Store results
            for quartile in metric_mean.index:
                file_results.append({
                    'file': file,
                    'metric': metric,
                    'quartile': quartile,
                    'mean_brier_score': metric_mean[quartile]
                })

# Convert file-specific results to a DataFrame
file_results_df = pd.DataFrame(file_results)

# Calculate the overall mean across all files for each metric and quartile
overall_results = file_results_df.groupby(['metric', 'quartile'])['mean_brier_score'].mean().reset_index()
print(overall_results)

# Set the Seaborn style
sns.set(style="whitegrid")

# Generate and save plots for each metric
for metric in metrics:
    metric_data = overall_results[overall_results['metric'] == metric]
    
    # Convert quartiles to descriptive names if needed
    quartiles = metric_data['quartile'].astype(str).tolist()
    scores = metric_data['mean_brier_score'].tolist()
    
    plt.figure(figsize=(10, 4))
    
    # Match colors based on the metric
    color = 'green' if metric == 'rr_1' else 'blue'
    
    sns.lineplot(
        x=quartiles, 
        y=scores, 
        marker='o', 
        linewidth=2.5, 
        label='Prior SOTA', 
        color=color
    )
    
    plt.xlabel('Quartiles', fontsize=12)
    plt.ylabel('Brier Score', fontsize=12)
    plt.title(f'{metric} Metric Performance by Quartile', fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'clmbr_metric_{metric}_rebuttal.png')
    plt.close()

print("Plots saved for all metrics with updated colors.")
