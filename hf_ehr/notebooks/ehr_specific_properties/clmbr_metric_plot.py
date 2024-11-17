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

# Initialize a list to store results for each file
file_results = []

# Process each file
for file in selected_files:
    file_path = os.path.join(csv_dir, file)
    if os.path.exists(file_path):
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Calculate mean for 'time_std' and 'ttr' per quartile
        time_std_data = data[data['strat_metric'] == 'time_std']
        ttr_data = data[data['strat_metric'] == 'ttr']
        
        time_std_mean = time_std_data.groupby('quartile')['brier_score'].mean()
        ttr_mean = ttr_data.groupby('quartile')['brier_score'].mean()
        
        # Store the results in the list
        for quartile in time_std_mean.index.union(ttr_mean.index):  # Handle quartiles in both metrics
            file_results.append({
                'file': file,
                'quartile': quartile,
                'mean_time_std': time_std_mean.get(quartile, None),
                'mean_ttr': ttr_mean.get(quartile, None)
            })

# Convert file-specific results to a DataFrame
file_results_df = pd.DataFrame(file_results)

# Calculate the overall mean across all files for each quartile
overall_results = file_results_df.groupby('quartile')[['mean_time_std', 'mean_ttr']].mean().reset_index()
print(overall_results)

# Data for CLMBR by quartiles
quartiles = ['Least', 'Less', 'More', 'Most']
std_time = list(overall_results['mean_time_std'])
trr = list(overall_results['mean_ttr'])

# Set the Seaborn style
sns.set(style="whitegrid")

# Create a Seaborn lineplot
plt.figure(figsize=(10, 4))

sns.lineplot(x=quartiles, y=std_time, label='Prior SOTA', marker='o', color='blue', linewidth=2.5)
#sns.lineplot(x=quartiles, y=trr, label='Prior SOTA', marker='o', color='green', linewidth=2.5)

# Add title and labels
#plt.title('Prior SOTA Model Performance v. EHR-Specific Properties', fontsize=15)
plt.xlabel('Repetitiveness', fontsize=12)
plt.ylabel('Brier Score', fontsize=12)

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('clmbr_metric_irr.png')
plt.close()
