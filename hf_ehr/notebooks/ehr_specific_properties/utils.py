import ast
import pickle
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import datetime
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from femr.labelers import LabeledPatients
from loguru import logger

# SPLITS
SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: int = 70
SPLIT_VAL_CUTOFF: int = 85

LABELING_FUNCTION_2_PAPER_NAME = {
    # Guo et al. 2023
    "guo_los": "Long LOS",
    "guo_readmission": "30-day Readmission",
    "guo_icu": "ICU Admission",
    # New diagnosis
    "new_pancan": "Pancreatic Cancer",
    "new_celiac": "Celiac",
    "new_lupus": "Lupus",
    "new_acutemi": "Acute MI",
    "new_hypertension": "Hypertension",
    "new_hyperlipidemia": "Hyperlipidemia",
    # Instant lab values
    "lab_thrombocytopenia": "Thrombocytopenia",
    "lab_hyperkalemia": "Hyperkalemia",
    "lab_hypoglycemia": "Hypoglycemia",
    "lab_hyponatremia": "Hyponatremia",
    "lab_anemia": "Anemia",
    # # Custom tasks
    # "chexpert": "Chest X-ray Findings",
    # # MIMIC-IV tasks
    # "mimic4_los" : "Long LOS (MIMIC-IV)",
    # "mimic4_readmission" : "30-day Readmission (MIMIC-IV)",
    # "mimic4_mortality" : "Inpatient Mortality (MIMIC-IV)",
}

TASK_GROUP_2_PAPER_NAME = {
    "operational_outcomes": "Operational Outcomes",
    "lab_values": "Anticipating Lab Test Results",
    "new_diagnoses": "Assignment of New Diagnoses",
    "chexpert": "Anticipating Chest X-ray Findings",
}

TASK_GROUP_2_LABELING_FUNCTION = {
    "operational_outcomes": [
        "guo_los",
        "guo_readmission",
        "guo_icu",
        "mimic4_los",
        "mimic4_mortality",
        "mimic4_readmission",
    ],
    "lab_values": [
        "lab_thrombocytopenia",
        "lab_hyperkalemia",
        "lab_hypoglycemia",
        "lab_hyponatremia",
        "lab_anemia"
    ],
    "new_diagnoses": [
        "new_hypertension",
        "new_hyperlipidemia",
        "new_pancan",
        "new_celiac",
        "new_lupus",
        "new_acutemi"
    ],
    "chexpert": [
        "chexpert"
    ],
}

def get_patient_splits_by_idx(path_to_split_csv: str, patient_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a list of patient IDs, split into train, val, and test sets.
        Returns the idxs for each split within `patient_ids`."""
    df_split = pd.read_csv(path_to_split_csv)
    split_2_idxs = { 'train' : [], 'val' : [], 'test' : [], }
    for split in ['train', 'val', 'test']:
        for idx, id in enumerate(patient_ids.tolist()):
            if id in df_split[df_split['split'] == split]['omop_person_id'].values:
                split_2_idxs[split].append(idx)
    return (
        split_2_idxs['train'],
        split_2_idxs['val'],
        split_2_idxs['test'],
    )

def compute_feature_label_alignment(label_pids, label_dates, feature_pids, feature_dates):
    result = np.zeros(label_pids.shape[0], dtype=np.uint32)
    j: int = 0
    for i in range(label_pids.shape[0]):
        while True:
            if j + 1 >= feature_pids.shape[0]:
                break
            elif feature_pids[j] < label_pids[i]:
                # Need to go ahead
                pass
            else:
                next_pid = feature_pids[j + 1]
                next_date = feature_dates[j + 1]

                if next_pid != label_pids[i]:
                    break

                if next_date > label_dates[i]:
                    break
            j += 1

        if feature_pids[j] != label_pids[i] or feature_dates[j] != label_dates[i]:
            raise RuntimeError(f"Could not find match for {label_pids[i]} {label_dates[i]}, closest is {feature_pids[j]} {feature_dates[j]}")
        result[i] = j
    return result

def get_labels_and_features(labeled_patients: LabeledPatients, 
                            path_to_features_dir: Optional[str], 
                            path_to_tokenized_timelines_dir: Optional[str],
                            models_to_keep: List[str] = []) -> Tuple[List[int], List[datetime.datetime], List[int], Dict[str, Dict[str, np.ndarray]]]:
    """Given a path to a directory containing labels and features as well as a LabeledPatients object, returns
        the labels and features for each patient. Note that this function is more complex b/c we need to align
        the labels with their corresponding features based on their prediction times."""
    label_patient_ids, label_values, label_times = labeled_patients.as_numpy_arrays()
    label_times = label_times.astype("datetime64[us]")

    # Sort arrays by (1) patient ID and (2) label time
    sort_order: np.ndarray = np.lexsort((label_times, label_patient_ids))
    label_patient_ids, label_values, label_times = label_patient_ids[sort_order], label_values[sort_order], label_times[sort_order]

    # Just return labels, ignore features
    if path_to_features_dir is None:
        return label_patient_ids, label_values, label_times

    # Go through every featurization we've created (e.g. count, clmbr, motor)
    # and align the label times with the featurization times
    featurizations: Dict[str, Dict[str, np.ndarray]] = {}
    for model in models_to_keep:
        print(f"Processing features for model: {model}")
        path_to_feats_file: str = os.path.join(path_to_features_dir, f'{model}_features.pkl')
        assert os.path.exists(path_to_feats_file), f'Path to file containing `{model}` features does not exist at this path: {path_to_feats_file}. Maybe you forgot to run `generate_features.py` first?'

        with open(path_to_feats_file, 'rb') as f:
            # Load data and do type checking
            feats = pickle.load(f)
            
            feature_tokenized_timelines = None
            if isinstance(feats, dict):
                if path_to_tokenized_timelines_dir is not None:
                    # HF_EHR format
                    path_to_tokenized_timelines_file: str = os.path.join(path_to_tokenized_timelines_dir, f'{model}_tokenized_timelines.npz')
                    assert os.path.exists(path_to_tokenized_timelines_file), f'Path to file containing `{model}` tokenized timelines does not exist at this path: {path_to_tokenized_timelines_file}. Maybe you forgot to run `generate_features.py` first?'
                    tokenized_timelines: np.ndarray = np.load(path_to_tokenized_timelines_file)['tokenized_timelines']
                    feature_matrix, feature_patient_ids, feature_times, feature_tokenized_timelines = (
                        feats['data_matrix'],
                        feats['patient_ids'],
                        feats['labeling_time'],
                        tokenized_timelines,
                    )
                    assert feature_tokenized_timelines.shape[0] == feature_matrix.shape[0], f'Error -- mismatched number of entries between feature_matrix={feature_matrix.shape[0]} and feature_tokenized_timelines={feature_tokenized_timelines.shape[0]}'
                else:
                    # CLMBR format
                    feature_matrix, feature_patient_ids, feature_times = (
                        feats['data_matrix'],
                        feats['patient_ids'],
                        feats['labeling_time'],
                    )
            else:
                # Count-based format
                feature_matrix, feature_patient_ids, feature_times = (
                    feats[0],
                    feats[1],
                    feats[3], # NOTE: skip label_values in [2]
                )

            feature_patient_ids = feature_patient_ids.astype(label_patient_ids.dtype)
            feature_times = feature_times.astype(label_times.dtype)
            assert feature_patient_ids.dtype == label_patient_ids.dtype, f'Error -- mismatched types between feature_patient_ids={feature_patient_ids.dtype} and label_patient_ids={label_patient_ids.dtype}'
            assert feature_times.dtype == label_times.dtype, f'Error -- mismatched types between feature_times={feature_times.dtype} and label_times={label_times.dtype}'

            # Sort arrays by (1) patient ID and (2) label time
            sort_order: np.ndarray = np.lexsort((feature_times, feature_patient_ids))
            feature_patient_ids, feature_times = feature_patient_ids[sort_order], feature_times[sort_order]

            # Align label times with feature times
            join_indices = compute_feature_label_alignment(label_patient_ids, 
                                                            label_times.astype(np.int64), 
                                                            feature_patient_ids, 
                                                            feature_times.astype(np.int64))
            feature_matrix = feature_matrix[sort_order[join_indices], :]
            if feature_tokenized_timelines is not None:
                feature_tokenized_timelines = feature_tokenized_timelines[sort_order[join_indices], :]

            # Validate that our alignment was successful
            assert np.all(feature_patient_ids[join_indices] == label_patient_ids)
            assert np.all(feature_times[join_indices] == label_times)

            featurizations[model] = {
                'frozen' : feature_matrix,
                'timelines' : feature_tokenized_timelines,
            }
    
    return label_patient_ids, label_values, label_times, featurizations

def process_chexpert_labels(label_values):
    new_labels = []
    for label_value in label_values:
        label_str = bin(label_value)[2:]
        rem_bin = 14 - len(label_str)
        label_str = "0"*rem_bin + label_str
        label_list = [*label_str]
        label_list = [int(label) for label in label_list]
        new_labels.append(label_list)
    return np.array(new_labels)

def convert_multiclass_to_binary_labels(values, threshold: int = 1):
    values[values >= threshold] = 1
    return values