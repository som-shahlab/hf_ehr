import pandas as pd
import os

# Define the folder to task name mapping
folder_to_name_mapping = {
    'guo_los': 'Long LOS',
    'guo_icu': 'ICU Prediction',
    'guo_readmission': '30-Day Readmission',
    'lab_anemia': 'Anemia',
    'lab_hyperkalemia': 'Hyperkalemia',
    'lab_hypoglycemia': 'Hypoglycemia',
    'lab_hyponatremia': 'Hyponatremia',
    'lab_thrombocytopenia': 'Thrombocytopenia',
    'new_acutemi': 'Acute MI',
    'new_celiac': 'Celiac',
    'new_hyperlipidemia': 'Hyperlipidemia',
    'new_hypertension': 'Hypertension',
    'new_lupus': 'Lupus',
    'new_pancan': 'Pancreatic Cancer',
}

# Function to clean data and tag 'att' models as a separate architecture
def clean_data_before_grouping(df):
    # Select and clean relevant columns, including the replicate column
    df = df[['sub_task', 'model', 'k', 'score', 'value', 'replicate']]
    df['model'] = df['model'].str.split('--clmbr_train').str[0]
    
    # Filter for AUROC scores
    df = df[(df['k'] == -1) & (df['score'] == 'auroc')]
    
    # Extract architecture and context length, tagging 'att' models
    df['architecture'] = df['model'].apply(lambda x: x.split('-')[0] + ('-att' if 'att' in x else ''))
    
    # Exclude BERT models
    df = df[df['architecture'] != 'bert']
    
    # Extract context length and ensure it's an integer
    df['context_length'] = df['model'].apply(lambda x: int(x.split('-')[2]) if len(x.split('-')) > 2 and x.split('-')[2].isdigit() else None)
    
    return df

# Function to compare att vs non-att models across all tasks
def compare_att_non_att_across_tasks(df):
    # Group by architecture, context_length, and sub_task to calculate the mean value (AUROC)
    df_grouped = df.groupby(['sub_task', 'architecture', 'context_length']).agg(
        value_mean=('value', 'mean')
    ).reset_index()

    # Pivot the data to compare att vs non-att
    comparison_df = df_grouped.pivot_table(
        index=['sub_task', 'context_length'],
        columns='architecture',
        values='value_mean'
    ).reset_index()

    # Identify columns for 'att' and non-'att' models
    att_cols = [col for col in comparison_df.columns if '-att' in col]
    non_att_cols = [col.replace('-att', '') for col in att_cols if col.replace('-att', '') in comparison_df.columns]

    # Calculate differences between 'att' and non-'att' models
    for att_col, non_att_col in zip(att_cols, non_att_cols):
        comparison_df[f'diff_{att_col}_vs_{non_att_col}'] = comparison_df[att_col] - comparison_df[non_att_col]

    return comparison_df

# Function to process all tasks for comparison
def process_all_tasks(ehrshot_dir, models_list, output_csv):
    combined_summary = pd.DataFrame()
    include_models = models_list
    exclude_tasks = []  # You can exclude specific tasks if needed

    # Iterate over the folder_to_name_mapping to process all relevant tasks
    for task_folder, task_name in folder_to_name_mapping.items():
        task_path = os.path.join(ehrshot_dir, task_folder)
        if os.path.isdir(task_path) and task_folder not in exclude_tasks:
            try:
                print(f"Processing task: {task_name}")
                input_file = os.path.join(task_path, 'all_results.csv')
                df = pd.read_csv(input_file)

                # Filter for relevant models
                df = df[df['model'].isin(include_models)]

                # Clean data and process 'att' tagging
                df_cleaned = clean_data_before_grouping(df)
                df_cleaned['task_name'] = task_name
                
                # Append to combined summary
                combined_summary = pd.concat([combined_summary, df_cleaned], ignore_index=True)

            except Exception as e:
                print(f"Failed to process task {task_name}: {e}")

    # Perform comparison between 'att' and non-'att' models
    comparison_df = compare_att_non_att_across_tasks(combined_summary)
    
    # Save the comparison to CSV
    comparison_df.to_csv(output_csv, index=False)
    print(f"Comparison saved to {output_csv}")

# Example usage
ehrshot_dir = '/share/pi/nigam/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/results_ehrshot'
models_list = [
    'mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last',
    'mamba-tiny-16384-att--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last']

output_csv = "att_vs_non_att_comparison_across_tasks.csv"

# Process all tasks and output comparison
process_all_tasks(ehrshot_dir, models_list, output_csv)
