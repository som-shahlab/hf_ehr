import argparse
import datetime
import os
import pickle
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import torch
from loguru import logger
import femr.datasets
from femr.labelers import LabeledPatients, load_labeled_patients
from hf_ehr.data.datasets import FEMRTokenizer
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.models.bert import BERTLanguageModel
from hf_ehr.models.hyena import HyenaLanguageModel
from hf_ehr.models.mamba import MambaLanguageModel
from hf_ehr.models.t5 import T5LanguageModel

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patient representations (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--path_to_model", type=str, help="Path to model .ckpt")
    parser.add_argument("--path_to_tokenizer", type=str, help="Path to tokenizer code_2_detail.json")
    parser.add_argument("--embed_strat", type=str, help="Strategy used for condensing a chunk of a timeline into a single embedding. Options: 'last' (only take last token), 'avg' (avg all tokens).")
    parser.add_argument("--chunk_strat", type=str, help="Strategy used for condensing a timeline longer than context window C. Options: 'last' (only take last chunk), 'avg' (avg all chunks together).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    EMBED_STRAT: str = args.embed_strat
    CHUNK_STRAT: str = args.chunk_strat
    IS_FORCE_REFRESH: bool = args.is_force_refresh
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    PATH_TO_MODEL = args.path_to_model
    PATH_TO_TOKENIZER_CODE_2_DETAIL = args.path_to_tokenizer
    batch_size: int = args.batch_size
    MODEL: str = args.path_to_model.split("/")[-3] 
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, f'{MODEL}_chunk:{CHUNK_STRAT}_embed:{EMBED_STRAT}_features.pkl')
    
    assert os.path.exists(PATH_TO_MODEL), f"No model exists @ `{PATH_TO_MODEL}`"

    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELED_PATIENTS}`")
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    
    logger.info(f"Loading PatientDatabase from `{PATH_TO_PATIENT_DATABASE}`")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE, read_all=True)
 
    feature_matrix, patient_ids, label_values, label_times = [], [], [], []
    device: str = args.device

    tokenizer = FEMRTokenizer(PATH_TO_TOKENIZER_CODE_2_DETAIL)
    vocab = tokenizer.get_vocab()
    
    checkpoint = torch.load(PATH_TO_MODEL, map_location='cpu')
    model_map = {
        'bert': BERTLanguageModel,
        'gpt2': GPTLanguageModel,
        'hyena': HyenaLanguageModel,
        'mamba': MambaLanguageModel,
        't5': T5LanguageModel
    }
    model_class = next((m for k, m in model_map.items() if k in MODEL), None)
    if not model_class:
        raise ValueError(f"Model `{MODEL}` not supported.")
    model = model_class(**checkpoint['hyper_parameters'], tokenizer=tokenizer)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    timeline_starts: Dict[int, List[datetime.datetime]] = {}
    timeline_tokens: Dict[int, List[int]] = {}
    for patient_id, labels in tqdm(labeled_patients.items(), desc="Loading EHRSHOT patient timelines"):
        full_timeline = [(x.start, x.code) for x in database[patient_id].events]
        timeline_with_valid_tokens = [x for x in full_timeline if x[1] in vocab]
        timeline_starts[patient_id] = [x[0] for x in timeline_with_valid_tokens]
        timeline_tokens[patient_id] = tokenizer([x[1] for x in timeline_with_valid_tokens])['input_ids'][0]
        assert len(timeline_starts[patient_id]) == len(timeline_tokens[patient_id]), f"Error - timeline_starts and timeline_tokens have different lengths for patient {patient_id}"

        for label in labels:
            patient_ids.append(patient_id)
            label_values.append(label.value)
            label_times.append(label.time)
    del database

    max_length: int = model.config.data.dataloader.max_length
    pad_token_id: int = tokenizer.token_2_idx['[PAD]']
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(patient_ids), batch_size), desc='Generating patient representations', total=len(patient_ids) // batch_size):
            pids = patient_ids[batch_start:batch_start + batch_size]
            values = label_values[batch_start:batch_start + batch_size]
            times = label_times[batch_start:batch_start + batch_size]
            timelines = []

            for pid, l_value, l_time in zip(pids, values, times):
                timeline_starts_for_pid = timeline_starts[pid]
                timeline_tokens_for_pid = timeline_tokens[pid]
                timeline = [token for start, token in zip(timeline_starts_for_pid, timeline_tokens_for_pid) if start <= l_time]
                timelines.append(timeline)
            
            if CHUNK_STRAT == 'last':
                timelines = [x[-max_length:] for x in timelines]
            else:
                raise ValueError(f"Chunk strategy `{CHUNK_STRAT}` not supported.")

            max_timeline_length = max(len(x) for x in timelines)
            timelines_w_pad = [[pad_token_id] * (max_timeline_length - len(x)) + x for x in timelines]
            input_ids = torch.stack([torch.tensor(x, device=device) for x in timelines_w_pad])
            attention_mask = (input_ids != pad_token_id).int()

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask if "hyena" not in MODEL else None
            }
            
            if 'hyena' in MODEL:
                batch.pop('attention_mask')

            results = model.model(**batch, output_hidden_states=True)
            hidden_states = results.hidden_states[-1]

            for idx in range(len(pids)):
                if EMBED_STRAT == 'last':
                    patient_rep = hidden_states[idx, -1, :].cpu().numpy()
                elif EMBED_STRAT == 'avg':
                    mask = input_ids[idx] != pad_token_id
                    patient_rep = hidden_states[idx, mask].mean(dim=0).cpu().numpy()
                else:
                    raise ValueError(f"Embedding strategy `{EMBED_STRAT}` not supported.")
                feature_matrix.append(patient_rep)

    feature_matrix = np.stack(feature_matrix)
    patient_ids = np.array(patient_ids)
    label_values = np.array(label_values)
    label_times = np.array(label_times)
    results = [feature_matrix, patient_ids, label_values, label_times]

    os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)
    logger.info(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")
    with open(PATH_TO_OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)

    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"label_values={repr(label_values)}\n"
                f"label_times={repr(label_times)}")
    logger.info(f"Shapes: feature_matrix={feature_matrix.shape}, patient_ids={patient_ids.shape}, label_values={label_values.shape}, label_times={label_times.shape}")
    logger.success("Done!")

if __name__ == "__main__":
    main()
