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
from hf_ehr.models.bert import BERTLanguageModel

'''
python3 ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_benchmark \
    --path_to_features_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_hf_features \
    --path_to_models_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models \
    --model gpt2-large-v8 \
    --embed_strat last \
    --chunk_strat last \
    --is_force_refresh
'''

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patient representations (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--path_to_models_dir", type=str, help="Path to directory where models are saved")
    parser.add_argument("--model", type=str, help="Name of foundation model to load.")
    parser.add_argument("--embed_strat", type=str, help="Strategy used for condensing a chunk of a timeline into a single embedding. Options: 'last' (only take last token), 'mean' (avg all tokens).")
    parser.add_argument("--chunk_strat", type=str, help="Strategy used for condensing a timeline longer than context window. Options: 'last' (only take last chunk), 'mean' (avg all chunks together).")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    # For larger models, need to create in chunks
    parser.add_argument("--start_idx", type=int, default=None, help="If set, then start repr generation at `start_idx`")
    return parser.parse_args()

def save_results(feature_matrix, patient_ids, label_values, label_times, path_to_output_file: str):
    results = [ feature_matrix, patient_ids, label_values, label_times ]

    # Save results
    os.makedirs(os.path.dirname(path_to_output_file), exist_ok=True)
    logger.info(f"Saving results to `{path_to_output_file}`")
    with open(path_to_output_file, 'wb') as f:
        pickle.dump(results, f)

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
    PATH_TO_TOKENIZER_CODE_2_INT = os.path.join(args.path_to_models_dir, MODEL, 'code_2_int.json')
    PATH_TO_TOKENIZER_CODE_2_COUNT = os.path.join(args.path_to_models_dir, MODEL, 'code_2_count.json')
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, f'{MODEL}_chunk:{CHUNK_STRAT}_embed:{EMBED_STRAT}_features.pkl')
    PATH_TO_MODEL = os.path.join(args.path_to_models_dir, MODEL, [file for file in os.listdir(os.path.join(args.path_to_models_dir, MODEL)) if file.endswith('.ckpt')][0])

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

    # Device
    device: str = 'cuda:0'
    
    # Load model
    checkpoint = torch.load(PATH_TO_MODEL)
    if 'bert' in MODEL:
        model = BERTLanguageModel(**checkpoint['hyper_parameters'])
    elif 'gpt2' in MODEL:
        model = GPTLanguageModel(**checkpoint['hyper_parameters'])
    else:
        raise ValueError(f"Model `{MODEL}` not supported.")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    # Load tokenizer
    atoi: Dict[str, int] = json.load(open(PATH_TO_TOKENIZER_CODE_2_INT, 'r'))
    code_to_count: Dict[str, int] = json.load(open(PATH_TO_TOKENIZER_CODE_2_COUNT, 'r'))
    min_code_count: Optional[int] = 10 if 'v8' in MODEL else None # TODO -- replace with model.config setting
    tokenizer = FEMRTokenizer(code_to_count, min_code_count=min_code_count)

    # Setup patient features
    timeline_starts: Dict[int, List[datetime.datetime]] = {}  # [key] = patient id, [value] = List of event.starts where [idx] is same as [idx] of corresponding code in `timeline_tokens`
    timeline_tokens: Dict[int, List[int]] =  {}  # [key] = patient id, [value] = List of event.codes where [idx] is same as [idx] of corresponding start in `timeline_starts`
    for patient_id, labels in tqdm(labeled_patients.items(), desc="Loading patient timelines"):
        # NOTE: Takes ~2 mins to load all patients
        # Create timeline for each label, where we only consider events that occurred BEFORE label.time
        full_timeline: List[Tuple[datetime.datetime, str]] = [ (x.start, x.code) for x in database[patient_id].events ]
        timeline_with_valid_tokens: List[Tuple[datetime.datetime, str]] = [ x for x in full_timeline if x[1] in tokenizer.vocab ]
        timeline_starts[patient_id] = [ x[0] for x in timeline_with_valid_tokens ]
        timeline_tokens[patient_id] = tokenizer([ x[1] for x in timeline_with_valid_tokens ])['input_ids'].squeeze(0).tolist()
        assert len(timeline_starts[patient_id]) == len(timeline_tokens[patient_id]), f"Error - timeline_starts and timeline_tokens have different lengths for patient {patient_id}"

        for label in labels:
            patient_ids.append(patient_id)
            label_values.append(label.value)
            label_times.append(label.time)

    # Generate patient representations
    max_length: int = model.config.data.dataloader.max_length
    with torch.no_grad():
        # NOTE: Takes ~5 hrs
        chunk_patient_ids, chunk_label_values, chunk_label_times = [], [], []
        for idx, (patient_id, label_value, label_time) in enumerate(tqdm(zip(patient_ids, label_values, label_times), desc='Generating patient representations', total=len(patient_ids))):
            # Get timeline
            if idx < args.start_idx:
                continue
            timeline_starts_for_patient: List[datetime.datetime] = timeline_starts[patient_id]
            timeline_tokens_for_patient: List[int] = timeline_tokens[patient_id]
            
            # Truncate timeline to only events <= label.time
            timeline: List[int] = []
            for token_idx, token in enumerate(timeline_tokens_for_patient):
                if timeline_starts_for_patient[token_idx] <= label_time:
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
            logits, hidden_states = model({ 'input_ids': torch.tensor([ timeline ]).to(device) })

            # Aggregate embeddings
            if EMBED_STRAT == 'last':
                patient_rep = hidden_states[:,-1,:].detach().cpu().numpy()
            elif EMBED_STRAT == 'mean':
                patient_rep = hidden_states.mean(dim=1).detach().cpu().numpy()
            else:
                raise ValueError(f"Embedding strategy `{EMBED_STRAT}` not supported.")
            feature_matrix.append(patient_rep)
            chunk_patient_ids.append(patient_id)
            chunk_label_values.append(label_value)
            chunk_label_times.append(label_time)
            
            # Need to save results every 100k patients to avoid memory/timeout issues
            if (idx + 1) % 100_000 == 0:
                path_to_output_chunk: str = PATH_TO_OUTPUT_FILE + "-{:03d}".format( ( idx + 1 ) // 100_000 )
                logger.info(f"Saving results to `{path_to_output_chunk}`")
                save_results(np.concatenate(feature_matrix), 
                             np.array(chunk_patient_ids), 
                             np.array(chunk_label_values), 
                             np.array(chunk_label_times), 
                             path_to_output_chunk)
                feature_matrix, chunk_patient_ids, chunk_label_values, chunk_label_times = [], [], [], []
    
    # Unify all chunks
    feature_matrix, patient_ids, label_values, label_times = [], [], [], []
    for file in os.listdir(os.path.dirname(PATH_TO_OUTPUT_FILE)):
        if file.startswith(os.path.basename(PATH_TO_OUTPUT_FILE)):
            with open(os.path.join(os.path.dirname(PATH_TO_OUTPUT_FILE), file), 'rb') as f:
                results = pickle.load(f)
                feature_matrix.append(results[0])
                patient_ids.append(results[1])
                label_values.append(results[2])
                label_times.append(results[3])

    save_results(np.concatenate(feature_matrix), 
                 np.concatenate(patient_ids), 
                 np.concatenate(label_values), 
                 np.concatenate(label_times), 
                 PATH_TO_OUTPUT_FILE)

    # Logging
    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"label_values={repr(label_values)}\n"
                f"label_times={repr(label_times)}")
    logger.info(f"Shapes: feature_matrix={feature_matrix.shape}, patient_ids={patient_ids.shape}, label_values={label_values.shape}, label_times={label_times.shape}")

    logger.success("Done!")
