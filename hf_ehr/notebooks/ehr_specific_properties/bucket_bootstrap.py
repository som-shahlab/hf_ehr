'''
Buckets EHRSHOT patients by a specific stratification metric and reports metrics for each bucket across all tasks and models.

Usage:
    python3 bucket_bootstrap.py
'''
import datetime
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import brier_score_loss
from utils import (
    get_labels_and_features, 
    get_patient_splits_by_idx
)
from femr.datasets import PatientDatabase
from femr.labelers import load_labeled_patients

np.random.seed(42)  

def calculate_brier_score_per_quartile(df, strat_column, strat_method, model, task, part=None):
    results = []
    if df.shape[0] != df.drop_duplicates().shape[0]:
        print('Dropping duplicates...')
        df.drop_duplicates(inplace=True)

    # Rank the strat_column and divide into quartiles
    df['quartile'] = pd.qcut(df[strat_column].rank(method='first'), 4, labels=False)

    for quartile in df['quartile'].unique():
        quartile_df = df[df['quartile'] == quartile]

        # Calculate Brier score for each quartile
        try:
            brier_score = brier_score_loss(quartile_df['y'], quartile_df['pred_proba'])
        except ValueError as e:
            print(f"Error calculating Brier score for quartile {quartile + 1}: {e}")
            continue

        results.append({
            'model': model,
            'task': task,
            'strat_method': strat_method,
            'strat_metric': strat_column if not part else f'{strat_column}_{part}',
            'quartile': f'{quartile + 1}',
            'brier_score': brier_score,
        })

    return results

def main():
    print("-" * 50)
    print("Computing stratified Brier score for all tasks and models.")
    print()

    # Constants
    PATH_TO_DATABASE = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract'
    PATH_TO_FEATURES_DIR = '/share/pi/nigam/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot'
    PATH_TO_RESULTS_DIR = '/share/pi/nigam/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/results_ehrshot'
    PATH_TO_TOKENIZED_TIMELINES_DIR = '/share/pi/nigam/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot'
    PATH_TO_LABELS_DIR = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot'
    PATH_TO_SPLIT_CSV = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv'

    # Output directory and master CSV path
    path_to_output_dir = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/'
    path_to_master_csv = os.path.join(path_to_output_dir, 'master_metrics_1000.csv')
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load existing master CSV if it exists
    if os.path.exists(path_to_master_csv):
        master_df = pd.read_csv(path_to_master_csv)
    else:
        master_df = pd.DataFrame()

    # Define the list of tasks and models you want to process
    tasks = [
        "guo_los", 
        "guo_readmission",
        "guo_icu",
        "new_hypertension",
        "new_hyperlipidemia",
        "new_pancan",
        "new_celiac",
        "new_lupus",
        "new_acutemi",
        "lab_thrombocytopenia",
        "lab_hyperkalemia",
        "lab_hypoglycemia",
        "lab_hyponatremia",
        "lab_anemia"
    ]
    models = [
        "gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last",
        "gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last",
        "hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last",
        "hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last",
        "llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last",
        "llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last",
        "mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last",
        "mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last"
    ]

    # Set the number of bootstraps
    N_BOOTSTRAPS = 1000  

    for task in tasks:
        for model in models:
            print(f"Processing task: {task}, model: {model}")
            # Load labeled patients for this task
            LABELING_FUNCTION = task
            PATH_TO_LABELED_PATIENTS = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
            femr_db = PatientDatabase(PATH_TO_DATABASE)
            labeled_patients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)

            # Get features for patients
            print(f"Processing features for model: {model}")
            patient_ids, label_values, label_times, _ = get_labels_and_features(
                labeled_patients, 
                PATH_TO_FEATURES_DIR, 
                PATH_TO_TOKENIZED_TIMELINES_DIR,
                models_to_keep=[model]
            )

            # Get patient splits
            train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)
            label_times = label_times[train_pids_idx + val_pids_idx + test_pids_idx]
            label_values = label_values[train_pids_idx + val_pids_idx + test_pids_idx]
            patient_ids = patient_ids[train_pids_idx + val_pids_idx + test_pids_idx]
            label_times = [x.astype(datetime.datetime) for x in label_times]  # cast to Python datetime
            assert len(train_pids_idx) + len(val_pids_idx) + len(test_pids_idx) == len(patient_ids)

            # Load EHRSHOT results
            head = 'lr_lbfgs'
            path_to_results_dir = os.path.join(PATH_TO_RESULTS_DIR, LABELING_FUNCTION, 'models')
            path_to_results_file = os.path.join(path_to_results_dir, 
                                                model, 
                                                head, 
                                                f'subtask={LABELING_FUNCTION}', 
                                                'k=-1',  # always use k=-1
                                                'preds.csv')
            if not os.path.exists(path_to_results_file):
                print(f'Path to results file does not exist: {path_to_results_file}. Skipping this model and task.')
                continue

            df_preds = pd.read_csv(path_to_results_file)
            df_preds['pid'] = patient_ids
            df_preds['label_time'] = label_times
            df_preds = df_preds.iloc[train_pids_idx + val_pids_idx + test_pids_idx]
            df_preds = df_preds[df_preds.index.isin(test_pids_idx)]
            if df_preds.shape[0] != len(test_pids_idx):
                print(f"Mismatch in number of test patients for model {model} and task {task}. Skipping.")
                continue

            for bootstrap in range(N_BOOTSTRAPS):
                print(f"Starting bootstrap {bootstrap + 1}/{N_BOOTSTRAPS} for task {task}, model {model}")
                # Randomly sample 1000 patients from the test set
                unique_test_pids = [patient_ids[idx] for idx in test_pids_idx]
                if len(unique_test_pids) < 1000:
                    print("Not enough patients in the test set to sample 1000 unique patients. Skipping.")
                    continue
                sampled_test_pids = np.random.choice(unique_test_pids, size=1000, replace=True)

                # Get df_preds for only those patients
                df_preds_sampled = df_preds[df_preds['pid'].isin(sampled_test_pids)].copy()

                strats = {
                    'inter_event_times': {
                        'strat_cols': ['time'],
                        'sep': 'metric'    
                    },
                    'n_gram_count': {
                        'strat_cols': ['rr_1']
                    }
                }

                for strat, metrics in tqdm(strats.items(), desc=f'Bootstrap {bootstrap + 1}/{N_BOOTSTRAPS} - Computing stratified Brier score'):
                    strat_cols = metrics['strat_cols']
                    sep = metrics.get('sep', None)
                    strat_metrics_file = os.path.join(path_to_output_dir, f'df__{task}__{strat}__metrics.parquet')
                    if not os.path.exists(strat_metrics_file):
                        print(f"Stratification metrics file not found: {strat_metrics_file}. Skipping this stratification.")
                        continue
                    df = pd.read_parquet(strat_metrics_file)
                    df_sampled = df[df['pid'].isin(sampled_test_pids)].copy()
                    for strat_col in strat_cols:
                        # Merge df_preds_sampled with df_sampled based on pid and label_time
                        if strat_col not in df.columns:
                            print(f'{strat_col} not in df columns. Skipping...')
                            continue
                        if sep:
                            for part in df_sampled['metric'].unique().tolist():
                                df2 = df_sampled[df_sampled['metric'] == part]
                                merged_df = pd.merge(df_preds_sampled, df2[['pid', 'label_time', strat_col]], how='left', on=['pid', 'label_time'])
                                # Ensure 'label_time' is datetime and consistent
                                merged_df['label_time'] = pd.to_datetime(merged_df['label_time'])
                                merged_df[strat_col] = pd.to_numeric(merged_df[strat_col], errors='coerce')
                                merged_df.dropna(subset=[strat_col], inplace=True)
                                if merged_df.empty:
                                    print(f"No data after merging for strat_col {strat_col} and part {part}. Skipping.")
                                    continue

                                strat_results = calculate_brier_score_per_quartile(
                                    merged_df, strat_col, strat, model, task, part
                                )
                                # Add 'bootstrap' index to results
                                for res in strat_results:
                                    res['bootstrap'] = bootstrap
                                master_df = pd.concat([master_df, pd.DataFrame(strat_results)], ignore_index=True)
                        else:
                            merged_df = pd.merge(df_preds_sampled, df_sampled[['pid', 'label_time', strat_col]], how='left', on=['pid', 'label_time'])
                            # Ensure 'label_time' is datetime and consistent
                            merged_df['label_time'] = pd.to_datetime(merged_df['label_time'])
                            merged_df[strat_col] = pd.to_numeric(merged_df[strat_col], errors='coerce')
                            merged_df.dropna(subset=[strat_col], inplace=True)
                            if merged_df.empty:
                                print(f"No data after merging for strat_col {strat_col}. Skipping.")
                                continue

                            strat_results = calculate_brier_score_per_quartile(
                                merged_df, strat_col, strat, model, task
                            )
                            # Add 'bootstrap' index to results
                            for res in strat_results:
                                res['bootstrap'] = bootstrap
                            master_df = pd.concat([master_df, pd.DataFrame(strat_results)], ignore_index=True)

            # Save the master DataFrame to CSV after each model-task combination
            master_df.to_csv(path_to_master_csv, index=False)
            print(f'Results saved to: {path_to_master_csv}')

    print('All tasks and models have been processed.')
    print('Done')

if __name__ == '__main__':
    main()
