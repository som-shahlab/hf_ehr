"""Usage:

python3 ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark \
    --path_to_features_dir / share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features \
    --path_to_model /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-base-clmbr/ckpts/epoch=1-step=150000-recent.ckpt \
    --embed_strat last \
    --chunk_strat last
"""

import argparse
import collections
import os
import pickle
import numpy as np
import torch
import json
import femr.datasets
from jaxtyping import Float
import shutil
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from tqdm import tqdm
from loguru import logger
from femr.labelers import LabeledPatients, load_labeled_patients
from hf_ehr.utils import load_config_from_path, load_tokenizer_from_path, load_model_from_path
from hf_ehr.config import Event

class CookbookModelWithClassificationHead(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, aggregation_strat: str, n_classes: int):
        super().__init__()
        self.n_classes: int = n_classes

        # Base model
        if model.model.__class__.__name__ == 'MambaForCausalLM':
            self.hidden_dim: int = model.model.lm_head.in_features
            self.base_model = model.model.backbone
            self.base_model_name = 'mamba'
        elif model.model.__class__.__name__ == 'GPT2LMHeadModel':
            self.hidden_dim: int = model.model.lm_head.in_features
            self.base_model = model.model.transformer
            self.base_model_name = 'gpt2'
        elif model.model.__class__.__name__ == 'LlamaForCausalLM':
            self.hidden_dim: int = model.model.lm_head.in_features
            self.base_model = model.model.model
            self.base_model_name = 'llama'
        elif model.model.__class__.__name__ in ['HyenaForCausalLM', 'HyenaDNAForCausalLM']:
            self.hidden_dim: int = model.model.lm_head.in_features
            self.base_model = model.model.hyena.backbone
            self.base_model_name = 'hyena'
        elif model.model.__class__.__name__ == 'BertForMaskedLM':
            self.hidden_dim: int = model.model.cls.predictions.transform.dense.in_features
            self.base_model = model.model.bert
            self.base_model_name = 'bert'
        else:
            raise ValueError(f"Unknown model type: {model.model.__class__.__name__}")

        # Aggregation of base model reprs for classification
        self.aggregation_strat = aggregation_strat

        # Linear head
        self.classifier = torch.nn.Linear(self.hidden_dim, n_classes, bias=True)

    def aggregate(self, x: Float[torch.Tensor, 'B L H']) -> Float[torch.Tensor, 'B H']:
        if self.aggregation_strat == 'mean':
            return torch.mean(x, dim=1)
        elif self.aggregation_strat == 'max':
            return torch.max(x, dim=1)
        elif self.aggregation_strat == 'last':
            return x[:, -1, :]
        elif self.aggregation_strat == 'first':
            return x[:, 0, :]
        else:
           raise ValueError(f"Aggregation strategy `{self.aggregation_strat}` not supported.") 

    def forward(self, *args, **kwargs) -> Float[torch.Tensor, 'B C']:
        """Return logits for classification task"""
        if self.base_model_name == 'hyena':
            reprs: Float[torch.Tensor, 'B L H'] = self.base_model(*args, **kwargs, output_hidden_states=True)[1][-1]
        else:
            reprs: Float[torch.Tensor, 'B L H'] = self.base_model(*args, **kwargs).last_hidden_state
        agg: Float[torch.Tensor, 'B H'] = self.aggregate(reprs)
        logits: Float[torch.Tensor, 'B C'] = self.classifier(agg)
        return logits
    
    def predict_proba(self, *args, **kwargs) -> Float[torch.Tensor, 'B C']:
        """Return a probability distribution over classes."""
        logits: Float[torch.Tensor, 'B C'] = self.forward(*args, **kwargs)
        return torch.softmax(logits, dim=-1)

    def predict(self, *args, **kwargs) -> Float[torch.Tensor, 'B']:
        """Return index of predicted class."""
        logits: Float[torch.Tensor, 'B C'] = self.forward(*args, **kwargs)
        return torch.argmax(logits, dim=-1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patient representations (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--path_to_tokenized_timelines_dir", required=True, type=str, help="Path to directory where tokenized timelines will be saved")
    parser.add_argument("--path_to_model", type=str, help="Path to model .ckpt")
    parser.add_argument("--model_name", type=str, default=None, help="If specified, replace folder name with this as the model's name")
    parser.add_argument("--embed_strat", type=str, help="Strategy used for condensing a chunk of a timeline into a single embedding. Options: 'last' (only take last token), 'mean' (avg all tokens).")
    parser.add_argument("--chunk_strat", type=str, help="Strategy used for condensing a timeline longer than context window C. Options: 'last' (only take last chunk), 'mean' (avg all chunks together).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    # For chunking
    parser.add_argument("--patient_idx_start", type=int, default=None, help="If specified, only process patients with idx >= this value (INCLUSIVE)")
    parser.add_argument("--patient_idx_end", type=int, default=None, help="If specified, only process patients with idx < this value (EXCLUSIVE)")
    return parser.parse_args()

def get_ckpt_name(path_to_ckpt: str) -> str:
    base_name = os.path.basename(path_to_ckpt)
    file_name, _ = os.path.splitext(base_name)
    return file_name

def process_in_batches(run_name, database, patient_ids, label_times, tokenizer, max_length, output_dir,
                      batch_size=1000, pad_token_id=0, chunk_strat='last') -> Dict[str, Any]:
    """
    Process patient events and create tokenized timelines in batches, saving each batch separately.
    Returns the list of batch files and metadata instead of loading everything into memory.
    
    Args:
        database: Database containing patient events
        patient_ids: List of patient IDs
        label_times: List of label times corresponding to patient IDs
        tokenizer: Tokenizer object
        max_length: Maximum length of tokenized timelines
        output_dir: Directory to save the batch files
        batch_size: Number of patients to process in each batch
        pad_token_id: Token ID used for padding
        chunk_strat: Strategy for chunking long sequences
    
    Returns:
        dict: Contains 'batch_files' list and metadata about the batches
    """
    logger.critical(f"Creating tokenized timelines from scratch @ `{output_dir}`")
    
    # Get total number of patients
    total_patients = len(patient_ids)
    
    # Create output directory for batch files
    os.makedirs(output_dir, exist_ok=True)
    
    batch_files = []
    batch_metadata = {
        'total_patients': total_patients,
        'max_length': max_length,
        'batch_size': batch_size,
        'batches': []
    }
    
    # Process patients in batches
    for batch_idx, batch_start in enumerate(tqdm(range(0, total_patients, batch_size), desc='Processing batches')):
        batch_end = min(batch_start + batch_size, total_patients)
        batch_patient_ids = patient_ids[batch_start:batch_end]
        total_batch = len(batch_patient_ids)
        batch_label_times = label_times[batch_start:batch_end]
        
        # Create batch-specific patient_id_2_events
        batch_patient_id_2_events: Dict[str, List[Event]] = collections.defaultdict(list)
        
        # Cache events for current batch of patients
        for pid in tqdm(batch_patient_ids, desc='Caching patient events', total=total_batch):
            if pid not in batch_patient_id_2_events:
                for e in database[pid].events:
                    batch_patient_id_2_events[pid].append(
                        Event(code=e.code, value=e.value, unit=e.unit, 
                              start=e.start, end=e.end, omop_table=e.omop_table)
                    )
        
        # Process each patient in the current batch
        batch_tokenized_timelines = []
        for pid, l_time in tqdm(zip(batch_patient_ids, batch_label_times), desc='Tokenizing timelines', total=total_batch):
            # Create patient timeline
            valid_events = [
                e for e in batch_patient_id_2_events[pid]
                if e.start <= l_time  # Ignore events after label time
            ]
            
            # Tokenize timeline
            timeline = tokenizer(valid_events, add_special_tokens=False)['input_ids'][0]
            
            # Apply chunking strategy
            if chunk_strat == 'last':
                timeline = timeline[-max_length:]
            else:
                raise ValueError(f"Chunk strategy `{chunk_strat}` not supported.")
            
            # PAD timeline to max_length
            padded_timeline = [pad_token_id] * (max_length - len(timeline)) + timeline  # left padding
            batch_tokenized_timelines.append(padded_timeline)
        
        # Convert batch results to numpy array
        batch_tokenized_timelines = np.array(batch_tokenized_timelines)
        
        # Save batch to file
        batch_file = os.path.join(output_dir, f'{run_name}_{batch_idx}.npz')
        np.savez_compressed(batch_file, 
                          tokenized_timelines=batch_tokenized_timelines,
                          start_idx=batch_start,
                          end_idx=batch_end)
        
        # Store batch metadata
        batch_metadata['batches'].append({
            'batch_id': f'{batch_idx}',
            'file': batch_file,
            'start_idx': int(batch_start),
            'end_idx': int(batch_end),
            'num_patients': int(batch_end - batch_start)
        })
        batch_files.append(batch_file)
        
        # Clear batch-specific memory
        del batch_patient_id_2_events
        del batch_tokenized_timelines
        
        logger.info(f"Saved batch {batch_idx} ({batch_start}-{batch_end}) to {batch_file}")

    # Save metadata file
    metadata_file = os.path.join(output_dir, f'{run_name}.json')
    with open(metadata_file, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
        logger.critical(f"Saved metadata file in: {metadata_file}")
    
    logger.critical(f"Processed {total_patients} patients in {len(batch_files)} batches")
    logger.critical(f"Batch files and metadata saved in: {output_dir}")
    
    return batch_metadata


def compute_feature_matrix(
    batch_metadata,
    config,
    model,
    patient_ids,
    batch_size,
    embed_strat,
    pad_token_id,
    device
)-> List:
    feature_matrix = []
    with torch.no_grad():
        
        for id, batch_dict in enumerate(batch_metadata['batches']):
            tokenized_timelines: List[List[int]] = np.load(batch_dict['file'])['tokenized_timelines'].tolist()

            for batch_start in tqdm(range(0, len(tokenized_timelines), batch_size), desc=f"Generating patient representations: {id}/{len(batch_metadata['batches'])}", total=len(tokenized_timelines) // batch_size):
                batch_end = min(len(tokenized_timelines), batch_start + batch_size)
                pids = patient_ids[batch_start:batch_end]
                batch_tokenized_timelines: List[List[int]] = tokenized_timelines[batch_start:batch_start + batch_size]

                ########################
                # Create batch
                ########################
                input_ids: Float[torch.Tensor, 'B max_timeline_length'] = torch.stack([torch.tensor(x, device=device) for x in batch_tokenized_timelines])
                attention_mask: Float[torch.Tensor, 'B max_timeline_length'] = (input_ids != pad_token_id).int()
                batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask if "hyena" not in config['model']['name'] else None
                }
                
                if 'hyena' in config['model']['name']:
                    batch.pop('attention_mask')

                ########################
                # Run model
                ########################
                results = model.model(**batch, output_hidden_states=True)
                hidden_states = results.hidden_states[-1]
                assert torch.isnan(hidden_states).sum() == 0, f"Error - hidden_states contains NaNs for batch_start={batch_start} | batch_end={batch_start + batch_size}"

                ########################
                # Save generated reprs
                ########################
                for idx in range(len(pids)):
                    if embed_strat == 'last':
                        try:
                            patient_rep = hidden_states[idx, -1, :].cpu().numpy()
                        except Exception as e:
                            breakpoint()
                    elif embed_strat == 'mean':
                        mask = input_ids[idx] != pad_token_id
                        patient_rep = hidden_states[idx, mask].mean(dim=0).cpu().numpy()
                    else:
                        raise ValueError(f"Embedding strategy `{embed_strat}` not supported.")
                    feature_matrix.append(patient_rep)
    return feature_matrix

def save_tokenized_timelines(npz_files: List, path_to_combined_timelines: str):
    # Load each .npz file and extract 'tokenized_timelines'
    all_timelines = []
    for file in npz_files:
        data = np.load(file)
        all_timelines.append(data['tokenized_timelines'])
    # Concatenate all arrays along the first axis
    combined_timelines = np.concatenate(all_timelines, axis=0)
    # Save the combined array into a single .npz file
    np.savez(path_to_combined_timelines, tokenized_timelines=combined_timelines)
    logger.critical(f"Saved combined timelines in: {path_to_combined_timelines}")

def main():
    args = parse_args()
    EMBED_STRAT: str = args.embed_strat
    CHUNK_STRAT: str = args.chunk_strat
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_TOKENIZED_TIMELINES_DIR: str = args.path_to_tokenized_timelines_dir
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    PATH_TO_MODEL = args.path_to_model
    MODEL: str = args.path_to_model.split("/")[-3] if args.model_name in [ None, '' ] else args.model_name
    CKPT: str = get_ckpt_name(PATH_TO_MODEL)
    batch_size: int = args.batch_size if args.batch_size not in [None, ''] else 16
    device: str = args.device
    patient_idx_start: Optional[int] = args.patient_idx_start
    patient_idx_end: Optional[int] = args.patient_idx_end
    model_signature: str = f'{MODEL}_{CKPT}_chunk:{CHUNK_STRAT}_embed:{EMBED_STRAT}'
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, model_signature)
    os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)
    assert os.path.exists(PATH_TO_MODEL), f"No model exists @ `{PATH_TO_MODEL}`"
    
    logger.critical(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")

    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELED_PATIENTS}`")
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    
    logger.info(f"Loading PatientDatabase from `{PATH_TO_PATIENT_DATABASE}`")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE, read_all=True)
    logger.info(f"Loading Config from `{PATH_TO_MODEL}`")
    config = load_config_from_path(PATH_TO_MODEL)
    logger.info(config)
    

    logger.info(f"Loading Tokenizer from `{PATH_TO_MODEL}")
    tokenizer = load_tokenizer_from_path(PATH_TO_MODEL)
    logger.info(f"Loading Model from `{PATH_TO_MODEL}`")
    model = load_model_from_path(PATH_TO_MODEL)
    model.to(device)
    model.eval()  # Set the model to evalevaluation mode
    # Filter patients by index (if specified)
    logger.info(f"Filtering patients by index: [{patient_idx_start}, {patient_idx_end})")
    allowed_pids = list(labeled_patients.keys())
    if patient_idx_start is not None and patient_idx_end is not None:
        allowed_pids = allowed_pids[patient_idx_start:patient_idx_end]
    elif patient_idx_start is not None:
        allowed_pids = allowed_pids[patient_idx_start:]
    elif patient_idx_end is not None:
        allowed_pids = allowed_pids[:patient_idx_end]
    else:
        allowed_pids = allowed_pids

    # Load all labels
    patient_ids, label_values, label_times = [], [], []
    for patient_id, labels in tqdm(labeled_patients.items(), desc=''):
        if allowed_pids is not None and patient_id not in allowed_pids:
            # Skip patients not in `allowed_pids`
            continue
        for label in labels:
            patient_ids.append(patient_id)
            label_values.append(label.value)
            label_times.append(label.time)
    logger.info(f"Total patient ids: {len(patient_ids)}")
    # Generate patient representations
    max_length: int = model.config.data.dataloader.max_length
    pad_token_id: int = tokenizer.token_2_idx['[PAD]']
    
    
    # Cache tokenized timelines for this sequence length
    run_name = f"chunk_strat={CHUNK_STRAT},max_length={max_length}_{config.data.tokenizer.name}" + (f'--start_idx={patient_idx_start}' if patient_idx_start else '') + (f'--end_idx={patient_idx_end}' if patient_idx_end else '') + "_tokenized_timelines"
    path_to_tokenized_timelines_metadata_file: str = os.path.join(PATH_TO_TOKENIZED_TIMELINES_DIR, f'{run_name}.json')
    if os.path.exists(path_to_tokenized_timelines_metadata_file):
        # Cache hit
        logger.success(f"Loading tokenized timelines from cache dir @ `{path_to_tokenized_timelines_metadata_file}`")
        with open(path_to_tokenized_timelines_metadata_file, 'r') as f:
            batch_metadata = json.load(f)
    else:
        batch_metadata = process_in_batches(
            database=database,
            patient_ids=patient_ids,
            label_times=label_times,
            tokenizer=tokenizer,
            max_length=max_length,
            output_dir=PATH_TO_TOKENIZED_TIMELINES_DIR,
            run_name=run_name,
            pad_token_id=pad_token_id,
            batch_size=8000
        )
    
    feature_matrix = compute_feature_matrix(
        batch_metadata,
        config,
        model,
        patient_ids,
        batch_size,
        EMBED_STRAT,
        pad_token_id,
        device
    )

    # Associate this featurization with its wandb run id + model path
    ## Save wandb run id of ckpt
    wandb_run_id = None
    path_to_wandb_txt: str = os.path.abspath(os.path.join(os.path.dirname(PATH_TO_MODEL), "../logs/wandb_run_id.txt"))
    if os.path.exists(path_to_wandb_txt):
        with open(path_to_wandb_txt, 'r') as f:
            wandb_run_id = f.read()
    else:
        logger.warning(f"No wandb run id found @ `{path_to_wandb_txt}`")
    ## Copy model ckpt .pt file over
    path_to_model_ehrshot_dir = os.path.abspath(os.path.join(PATH_TO_FEATURES_DIR, "../models/", os.path.basename(PATH_TO_OUTPUT_FILE)))
    logger.critical(f"Copying model ckpt from `{PATH_TO_MODEL}` to `{path_to_model_ehrshot_dir}`")
    if os.path.exists(path_to_model_ehrshot_dir):
        shutil.rmtree(path_to_model_ehrshot_dir)
    os.makedirs(path_to_model_ehrshot_dir, exist_ok=True)
    shutil.copy(PATH_TO_MODEL, path_to_model_ehrshot_dir)
    ## Copy logging files over
    logger.critical(f"Copying logs from `{os.path.join(os.path.dirname(PATH_TO_MODEL), '../logs/')}` to `{path_to_model_ehrshot_dir}/logs`")
    # shutil.copytree(os.path.join(os.path.dirname(PATH_TO_MODEL), '../logs/'), os.path.join(path_to_model_ehrshot_dir, 'logs/'))
    ## Copy tokenized timelines over into EHRSHOT cache dir and associate with this model
    signature = model_signature + (f'--start_idx={patient_idx_start}' if patient_idx_start else '') + (f'--end_idx={patient_idx_end}' if patient_idx_end else '') + f"_tokenized_timelines.npz"
    path_to_tokenized_timelines_ehrshot_file: str = os.path.join(PATH_TO_TOKENIZED_TIMELINES_DIR, signature)
    logger.critical(f"Copying tokenized timelines from to `{path_to_tokenized_timelines_ehrshot_file}`")
    save_tokenized_timelines([metadata['file'] for metadata in batch_metadata['batches']], path_to_tokenized_timelines_ehrshot_file)
    # Save EHRSHOT featurization results
    logger.info(f"Stacking featurization results...")
    feature_matrix = np.stack(feature_matrix)
    patient_ids = np.array(patient_ids)
    label_values = np.array(label_values)
    label_times = np.array(label_times)
    assert label_values.shape == label_times.shape, f"Error - label_values and label_times have different shapes: {label_values.shape} vs {label_times.shape}"
    assert label_values.shape == patient_ids.shape, f"Error - label_values and patient_ids have different shapes: {label_values.shape} vs {patient_ids.shape}"
    assert feature_matrix.shape[0] == batch_metadata['total_patients'], f"Error - feature_matrix and tokenized_timelines have different lengths: {feature_matrix.shape[0]} vs {batch_metadata['total_patients']}"
    results = {
        'data_matrix' : feature_matrix, # frozen features from model
        'patient_ids' : patient_ids,
        'labeling_time' : label_times,
        'label_values' : label_values,
        'wandb_run_id' : wandb_run_id,
        'path_to_ckpt_ehrshot' : path_to_model_ehrshot_dir,
        'path_to_ckpt_orig' : PATH_TO_MODEL,
    }

    path_to_features_pkl: str = PATH_TO_OUTPUT_FILE + (f'--start_idx={patient_idx_start}' if patient_idx_start else '') + (f'--end_idx={patient_idx_end}' if patient_idx_end else '') + '_features_1.pkl'
    logger.critical(f"Saving results to `{path_to_features_pkl}`")
    with open(path_to_features_pkl, 'wb') as f:
        pickle.dump(results, f)

    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids[:10])}\n"
                f"label_values={repr(label_values[:10])}\n"
                f"label_times={repr(label_times[:10])}")
    logger.info(f"Shapes: feature_matrix={feature_matrix.shape}, patient_ids={patient_ids.shape}, label_values={label_values.shape}, label_times={label_times.shape}")
    logger.success("Done!")

if __name__ == "__main__":
    main()
