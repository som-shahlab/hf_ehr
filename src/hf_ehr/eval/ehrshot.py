import argparse
import datetime
import os
import pickle
import json

from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import numpy as np

import torch
from loguru import logger
import femr.datasets
from femr.labelers import LabeledPatients, load_labeled_patients
from hf_ehr.data.datasets import FEMRTokenizer
from hf_ehr.models.gpt import GPTLanguageModel

'''
python3 ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_benchmark \
    --path_to_features_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_hf_features \
    --path_to_models_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models \
    --model gpt2-base \
    --embed_strat last \
    --chunk_strat last \
    --is_force_refresh
'''

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLMBR / MOTOR patient representations (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--path_to_models_dir", type=str, help="Path to directory where models are saved")
    parser.add_argument("--model", type=str, help="Name of foundation model to load.")
    parser.add_argument("--embed_strat", type=str, help="Strategy used for condensing a chunk of a timeline into a single embedding. Options: 'last' (only take last token), 'avg' (avg all tokens).")
    parser.add_argument("--chunk_strat", type=str, help="Strategy used for condensing a timeline longer than context window C. Options: 'last' (only take last chunk), 'avg' (avg all chunks together).")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL: str = args.model
    EMBED_STRAT: str = args.embed_strat
    CHUNK_STRAT: str = args.chunk_strat
    IS_FORCE_REFRESH: bool = args.is_force_refresh
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    PATH_TO_MODEL = os.path.join(args.path_to_models_dir, MODEL)
    PATH_TO_TOKENIZER = os.path.join(args.path_to_models_dir, MODEL, 'code_2_int.json')
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, f'{MODEL}_features.pkl')
    
    # Check that requested model exists
    assert os.path.exists(PATH_TO_MODEL), f"No model for `{MODEL}` exists @ `{PATH_TO_MODEL}`"

    # Load consolidated labels across all patients for all tasks
    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELED_PATIENTS}`")
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    
    # FEMR database
    logger.info(f"Loading PatientDatabase from `{PATH_TO_PATIENT_DATABASE}`")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE, read_all=True)
 
    # Run inference on every patient
    feature_matrix, patient_ids, label_values, label_times = [], [], [], []

    # Load model
    checkpoint = torch.load('/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models/gpt2-base/epoch=11-step=479000-val_loss.ckpt')
    model = GPTLanguageModel(**checkpoint['hyper_parameters'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda:0')
    
    # Load tokenizer
    atoi: Dict[str, int] = json.load(open(PATH_TO_TOKENIZER, 'r'))
    tokenizer = FEMRTokenizer(atoi)

    # Setup patient features
    timeline_starts: Dict[int, List[datetime.datetime]] = {}  # [key] = patient id, [value] = List of event.starts where [idx] is same as [idx] of corresponding code in `timeline_tokens`
    timeline_tokens: Dict[int, List[int]] =  {}  # [key] = patient id, [value] = List of event.codes where [idx] is same as [idx] of corresponding start in `timeline_starts`
    for patient_id, labels in tqdm(labeled_patients.items(), desc="Loading patient timelines"):
        # NOTE: Takes ~2 mins to load all patients
        # Create timeline for each label, where we only consider events that occurred BEFORE label.time
        full_timeline: List[Tuple[datetime.datetime, str]] = [ (x.start, x.code) for x in database[patient_id].events ]
        timeline_starts[patient_id] = [ x[0] for x in full_timeline ]
        timeline_tokens[patient_id] = tokenizer.tokenize([ x[1] for x in full_timeline if x[1] in tokenizer.atoi ], add_special_tokens=True)['input_ids'].squeeze(0).tolist()
        for label in labels:
            patient_ids.append(patient_id)
            label_values.append(label.value)
            label_times.append(label.time)

    # Generate patient representations
    max_length: int = model.config.data.dataloader.max_length
    with torch.no_grad():
        for (patient_id, label_value, label_time) in tqdm(zip(patient_ids, label_values, label_times), desc='Generating patient representations', total=len(patient_ids)):
            # Get timeline
            timeline_starts_for_patient: List[datetime.datetime] = timeline_starts[patient_id]
            timeline_tokens_for_patient: List[int] = timeline_tokens[patient_id]
            
            # Truncate timeline to only events < label.time
            timeline: List[int] = []
            for token_idx, token in enumerate(timeline_tokens_for_patient):
                if timeline_starts_for_patient[token_idx] < label_time:
                    timeline.append(token)
                else:
                    break

            # Chunking
            if CHUNK_STRAT == 'last':
                timeline = timeline[-max_length:]
            else:
                raise ValueError(f"Chunk strategy `{CHUNK_STRAT}` not supported.")

            # Inference
            # logits.shape = (batch_size = 1, sequence_length, vocab_size = 167k)
            # hidden_states.shape = (batch_size = 1, sequence_length, hidden_size = 768)
            logits, hidden_states = model({ 'input_ids': torch.tensor([ timeline ]).to('cuda:0') })

            # Aggregate embeddings
            if EMBED_STRAT == 'last':
                patient_rep = hidden_states[:,-1,:]
            elif EMBED_STRAT == 'avg':
                patient_rep = hidden_states.mean(dim=1)
            else:
                raise ValueError(f"Embedding strategy `{EMBED_STRAT}` not supported.")
            feature_matrix.append(patient_rep)
    
    feature_matrix = torch.cat(feature_matrix, dim=0).cpu().numpy()
    patient_ids = np.array(patient_ids)
    label_values = np.array(label_values)
    label_times = np.array(label_times)
    results = [ feature_matrix, patient_ids, label_values, label_times ]

    # Save results
    os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)
    logger.info(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")
    with open(PATH_TO_OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)

    # Logging
    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"label_values={repr(label_values)}\n"
                f"label_times={repr(label_times)}")

    logger.success("Done!")
