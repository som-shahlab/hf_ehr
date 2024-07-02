import argparse
import datetime
import os
import pickle
import json

from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf

import torch
from loguru import logger
import femr.datasets
from femr.labelers import LabeledPatients, load_labeled_patients
from hf_ehr.data.datasets import FEMRTokenizer
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.models.bert import BERTLanguageModel
from hf_ehr.models.hyena import HyenaLanguageModel
from hf_ehr.models.mamba import MambaLanguageModel

'''
python3 ehrshot.py \
    --path_to_database /local-scratch-nvme/nigam/ehrshot/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /local-scratch-nvme/nigam/ehrshot/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark \
    --path_to_features_dir /local-scratch-nvme/nigam/ehrshot/ehrshot-benchmark/EHRSHOT_ASSETS/features \
    --path_to_model /share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt2-base-lr-1e-4/ckpts/last.ckpt \
    --path_to_tokenizer /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_detail.json \
    --embed_strat last \
    --chunk_strat last \
    --is_force_refresh
    
    
python3 ehrshot.py \
    --path_to_database /local-scratch-nvme/nigam/ehrshot/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /local-scratch-nvme/nigam/ehrshot/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark \
    --path_to_features_dir /local-scratch-nvme/nigam/ehrshot/ehrshot-benchmark/EHRSHOT_ASSETS/features \
    --path_to_model /share/pi/nigam/suhana/hf_ehr/cache/runs/mamba_tiny_16_1e6/ckpts/last-v1.ckpt \
    --path_to_tokenizer /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_detail.json \
    --embed_strat last \
    --chunk_strat last \
    --is_force_refresh
'''

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
    MODEL: str = args.path_to_model.split("/")[-3] # /share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt2-base-lr-1e-4/ckpts/last.ckpt => gpt2-base-lr-1e-4
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, f'{MODEL}_chunk:{CHUNK_STRAT}_embed:{EMBED_STRAT}_features.pkl')
    
    # Check that requested model exists
    assert os.path.exists(PATH_TO_MODEL), f"No model exists @ `{PATH_TO_MODEL}`"

    # Load consolidated labels across all patients for all tasks
    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELED_PATIENTS}`")
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    
    # FEMR database
    logger.info(f"Loading PatientDatabase from `{PATH_TO_PATIENT_DATABASE}`")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE, read_all=True)
 
    # Run inference on every patient
    feature_matrix, patient_ids, label_values, label_times = [], [], [], []

    # Device
    device: str = args.device

    # Load tokenizer
    tokenizer = FEMRTokenizer(PATH_TO_TOKENIZER_CODE_2_DETAIL)
    vocab = tokenizer.get_vocab()
    
    # Load model
    checkpoint = torch.load(PATH_TO_MODEL, map_location='cpu')
    if 'bert' in MODEL:
        model = BERTLanguageModel(**checkpoint['hyper_parameters'], tokenizer=tokenizer)
    elif 'gpt2' in MODEL:
        model = GPTLanguageModel(**checkpoint['hyper_parameters'], tokenizer=tokenizer)
    elif 'hyena' in MODEL:
        model = HyenaLanguageModel(**checkpoint['hyper_parameters'], tokenizer=tokenizer)
    elif 'mamba' in MODEL:
        model = MambaLanguageModel(**checkpoint['hyper_parameters'], tokenizer=tokenizer)
    else:
        raise ValueError(f"Model `{MODEL}` not supported.")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    # Setup patient features
    timeline_starts: Dict[int, List[datetime.datetime]] = {}  # [key] = patient id, [value] = List of event.starts where [idx] is same as [idx] of corresponding code in `timeline_tokens`
    timeline_tokens: Dict[int, List[int]] =  {}  # [key] = patient id, [value] = List of event.codes where [idx] is same as [idx] of corresponding start in `timeline_starts`
    count = 0
    for patient_id, labels in tqdm(labeled_patients.items(), desc="Loading EHRSHOT patient timelines"):
        # NOTE: Takes ~2 mins to load all patients
        # Create timeline for each label, where we only consider events that occurred BEFORE label.time
        full_timeline: List[Tuple[datetime.datetime, str]] = [ (x.start, x.code) for x in database[patient_id].events ]
        timeline_with_valid_tokens: List[Tuple[datetime.datetime, str]] = [ x for x in full_timeline if x[1] in vocab ]
        timeline_starts[patient_id] = [ x[0] for x in timeline_with_valid_tokens ]
        timeline_tokens[patient_id] = tokenizer([ x[1] for x in timeline_with_valid_tokens ])['input_ids'][0]
        assert len(timeline_starts[patient_id]) == len(timeline_tokens[patient_id]), f"Error - timeline_starts and timeline_tokens have different lengths for patient {patient_id}"

        for label in labels:
            patient_ids.append(patient_id)
            label_values.append(label.value)
            label_times.append(label.time)
        
        count += 1
        # if count == 20:
        #     break
    
    # Generate patient representations
    max_length: int = model.config.data.dataloader.max_length
    with torch.no_grad():
        # NOTE: Takes ~5 hrs
        chunk_patient_ids, chunk_label_values, chunk_label_times = [], [], []
        for idx, batch_start in enumerate(tqdm(range(0, len(patient_ids), batch_size), desc='Generating patient representations', total=len(patient_ids) // batch_size)):
            if args.start_idx and idx < args.start_idx:
                continue
            
            pids = patient_ids[batch_start:batch_start + batch_size]
            values = label_values[batch_start:batch_start + batch_size]
            times = label_times[batch_start:batch_start + batch_size]
            timelines: List[List[int]] = []

            # Truncate timeline to only events <= label.time
            for (pid, l_value, l_time) in zip(pids, values, times):
                timeline_starts_for_pid: List[datetime.datetime] = timeline_starts[pid]
                timeline_tokens_for_pid: List[int] = timeline_tokens[pid]
                timeline: List[int] = []
                for token_idx, token in enumerate(timeline_tokens_for_pid):
                    if timeline_starts_for_pid[token_idx] <= l_time:
                        timeline.append(token)
                    else:
                        break
                timelines.append(timeline)
            
            # Chunking
            if CHUNK_STRAT == 'last':
                timelines = [ x[-max_length:] for x in timelines ]
            else:
                raise ValueError(f"Chunk strategy `{CHUNK_STRAT}` not supported.")

            max_timeline_length = max([ len(x) for x in timelines ])

            # Left PAD timelines to max_timeline_length
            pad_token_id: int = tokenizer.token_2_idx['[PAD]']
            timelines_w_pad: List[List[int]] = [ [ pad_token_id ] * (max_timeline_length - len(x)) + x  for x in timelines ]
            assert all([ len(x) == max_timeline_length for x in timelines_w_pad ])
            
            # Creates a tensor of size: batch_size x max_timeline_length
            input_ids: torch.Tensor = torch.stack([ torch.tensor(x) for x in timelines_w_pad ])
            batch = {
                'input_ids' : input_ids.to(device),
                'attention_mask' : (input_ids != pad_token_id).int().to(device),
            }

            # Inference
            # logits.shape = (batch_size, sequence_length, vocab_size = 167k)
            # hidden_states.shape = (batch_size, sequence_length, hidden_size = 768)
            results = model.model(**batch, output_hidden_states=True)
            hidden_states = results.hidden_states[-1].detach().cpu()

            # Aggregate embeddings
            if EMBED_STRAT == 'last':
                for pidx in range(0, len(pids)):
                    patient_rep = hidden_states[pidx,-1,:].numpy()
                    feature_matrix.append(patient_rep)
            elif EMBED_STRAT == 'avg':
                for pidx in range(0, len(pids)):
                    # NOTE: Need to account for attention_mask when averaging over tokens to ignore [PAD]
                    patient_rep = hidden_states[pidx, input_ids[pidx] != pad_token_id].mean(dim=1).numpy()
                    feature_matrix.append(patient_rep)
            else:
                raise ValueError(f"Embedding strategy `{EMBED_STRAT}` not supported.")

            chunk_patient_ids.extend(pids)
            chunk_label_values.extend(values)
            chunk_label_times.extend(times)
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} batches")
                path_to_output_chunk: str = PATH_TO_OUTPUT_FILE + "-{:03d}".format( ( idx + 1 ) // 1000 )
                logger.info(f"Saving results to `{path_to_output_chunk}`")
                save_results(np.stack(feature_matrix), 
                             np.array(chunk_patient_ids), 
                             np.array(chunk_label_values), 
                             np.array(chunk_label_times), 
                             path_to_output_chunk)
                feature_matrix, chunk_patient_ids, chunk_label_values, chunk_label_times = [], [], [], []
                # break
      
      
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
                print(feature_matrix)

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
    # logger.info(f"Shapes: feature_matrix={feature_matrix.shape}, patient_ids={patient_ids.shape}, label_values={label_values.shape}, label_times={label_times.shape}")

    logger.success("Done!")
    
    # feature_matrix = np.stack(feature_matrix)
    # patient_ids = np.array(patient_ids)
    # label_values = np.array(label_values)
    # label_times = np.array(label_times)
    # results = [ feature_matrix, patient_ids, label_values, label_times ]

    # # Save results
    # os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)
    # logger.info(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")
    # with open(PATH_TO_OUTPUT_FILE, 'wb') as f:
    #     pickle.dump(results, f)

    # # Logging
    # logger.info("FeaturizedPatient stats:\n"
    #             f"feature_matrix={repr(feature_matrix)}\n"
    #             f"patient_ids={repr(patient_ids)}\n"
    #             f"label_values={repr(label_values)}\n"
    #             f"label_times={repr(label_times)}")
    # logger.info(f"Shapes: feature_matrix={feature_matrix.shape}, patient_ids={patient_ids.shape}, label_values={label_values.shape}, label_times={label_times.shape}")

    # logger.success("Done!")

