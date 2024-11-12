'''
Buckets EHRSHOT patients by a specific stratification metric

Usage:
    python3 bucket.py
'''
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from utils import (
    get_labels_and_features, 
    get_patient_splits_by_idx
)
from femr.labelers import load_labeled_patients, LabeledPatients

PATH_TO_FEATURES_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot'
PATH_TO_TOKENIZED_TIMELINES_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot'
PATH_TO_LABELS_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot'
PATH_TO_SPLIT_CSV: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv'

strats = {
    'inter_event_times': [ 'std' ],
    'n_gram_count': ['rr_1'], 
    'timeline_lengths': ['n_tokens'],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='EHRSHOT task', default='guo_los')
    args = parser.parse_args()
    
    # Output directory
    path_to_output_dir: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/'
    
    # Load labeled patients for this task
    LABELING_FUNCTION: str = args.task
    PATH_TO_LABELED_PATIENTS: str =  os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)

    # Load labels for this task
    default_model: str = 'gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last'
    patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(labeled_patients, 
                                                                                        PATH_TO_FEATURES_DIR, 
                                                                                        PATH_TO_TOKENIZED_TIMELINES_DIR,
                                                                                        models_to_keep=[default_model,])
    train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)
    label_times = [ x.astype(datetime.datetime) for x in label_times ] # cast to Python datetime
    assert len(train_pids_idx) + len(val_pids_idx) + len(test_pids_idx) == len(patient_ids)
    assert len(np.intersect1d(train_pids_idx, val_pids_idx)) == 0
    assert len(np.intersect1d(train_pids_idx, test_pids_idx)) == 0
    assert len(np.intersect1d(val_pids_idx, test_pids_idx)) == 0
    
    # Calculate quartiles based on each stratification metric
    df_results = []
    for strat, strat_cols in tqdm(strats.items(), desc=f'Stratifying {LABELING_FUNCTION}'):
        # Load metric for each patient
        df_metrics = pd.read_parquet(os.path.join(path_to_output_dir, f'df__{LABELING_FUNCTION}__{strat}__metrics.parquet'))
        
        # For every metric, calculate quartiles
        for strat_col in strat_cols:
            # If stratifying by inter-event times, need to pivot table since 'time' and 'metric' are separate columns
            if strat == 'inter_event_times':
                df_metrics = df_metrics.pivot_table(index=['pid', 'pid_idx', 'label_time', 'sub_task'], columns='metric', values='time').reset_index()

            if strat_col not in df_metrics.columns:
                raise ValueError(f'col={strat_col} not in df_metrics columns for strat={strat}.')

            # Calculate quartiles
            df_metrics['metric_name'] = f'{strat_col}'
            df_metrics['quartile'] = pd.qcut(df_metrics[strat_col].rank(method='min'), 4, labels=False)
            df_metrics = df_metrics.rename(columns={strat_col: 'metric_value'})
            
            df_results.append(df_metrics)
            
    path_to_output_file: str = os.path.join(path_to_output_dir, f'df_bucket__{LABELING_FUNCTION}.csv')
    print(f"Saving results to {path_to_output_file}")
    df_results = pd.concat(df_results)
    df_results.to_csv(path_to_output_file, index=False)

    print('Done!')