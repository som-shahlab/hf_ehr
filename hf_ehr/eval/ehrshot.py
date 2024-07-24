"""Usage:

python3 ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark \
    --path_to_features_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features \
    --path_to_model /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-base-clmbr/ckpts/epoch=1-step=150000-recent.ckpt \
    --embed_strat last \
    --chunk_strat last
"""

import argparse
import datetime
import os
import json
import pickle
import numpy as np
import torch
import femr.datasets
from jaxtyping import Float
import shutil
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from tqdm import tqdm
from loguru import logger
from femr.labelers import LabeledPatients, load_labeled_patients
from hf_ehr.data.datasets_old import convert_event_to_token
from hf_ehr.utils import load_config_from_path, load_tokenizer_from_path, load_model_from_path, load_tokenizer_from_config


class CookbookModelWithClassificationHead(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, aggregation_strat: str, n_classes: int):
        super().__init__()
        self.n_classes: int = n_classes
        self.hidden_dim: int = model.model.lm_head.in_features

        # Base model
        if model.model.__class__.__name__ == 'MambaForCausalLM':
            self.base_model = model.model.backbone
            self.base_model_name = 'mamba'
        elif model.model.__class__.__name__ == 'GPT2LMHeadModel':
            self.base_model = model.model.transformer
            self.base_model_name = 'gpt2'
        elif model.model.__class__.__name__ == 'HyenaForCausalLM':
            self.base_model = model.model.hyena.backbone
            self.base_model_name = 'hyena'
        elif model.model.__class__.__name__ == 'BertForMaskedLM':
            self.base_model = model.model.bert.encoder
            self.base_model_name = 'bert'
        else:
            raise ValueError("Model must be a MambaForCausalLM")

        # Aggregation of base model reprs for classification
        self.aggregation_strat = aggregation_strat

        # Linear head
        self.classifier = torch.nn.Linear(self.hidden_dim, n_classes)

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

    def forward(self, input_ids: Float[torch.Tensor, 'B L'] = None, attention_mask: Float[torch.Tensor, 'B L'] = None, **kwargs) -> Float[torch.Tensor, 'B C']:
        """Return logits for classification task"""
        reprs: Float[torch.Tensor, 'B L H'] = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        agg: Float[torch.Tensor, 'B H'] = self.aggregate(reprs)
        logits: Float[torch.Tensor, 'B C'] = self.classifier(agg)
        return logits
    
    def predict_proba(self, input_ids: Float[torch.Tensor, 'B L'] = None, attention_mask: Float[torch.Tensor, 'B L'] = None, **kwargs) -> Float[torch.Tensor, 'B C']:
        """Return a probability distribution over classes."""
        logits: Float[torch.Tensor, 'B C'] = self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return torch.softmax(logits, dim=-1)

    def predict(self, input_ids: Float[torch.Tensor, 'B L'] = None, attention_mask: Float[torch.Tensor, 'B L'] = None, **kwargs) -> Float[torch.Tensor, 'B']:
        """Return index of predicted class."""
        logits: Float[torch.Tensor, 'B C'] = self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return torch.argmax(logits, dim=-1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patient representations (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
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
    
def generate_ehrshot_timelines(database: femr.datasets.PatientDatabase,
                               config,
                               labeled_patients: LabeledPatients, 
                               allowed_pids: Optional[Union[Set[int], List[int]]] = None,
                               tqdm_desc: str = "Loading EHRSHOT patient timelines") -> Dict[str, Any]:
    """Generate patient timelines for EHRSHOT task."""
    # Load config
    tokenizer__excluded_vocabs: Optional[List[str]] = config.data.tokenizer.excluded_vocabs
    tokenizer__min_code_count: Optional[int] = getattr(config.data.tokenizer, 'min_code_count', None)
    tokenizer__is_remap_numerical_codes: bool = getattr(config.data.tokenizer, 'is_remap_numerical_codes', False)
    tokenizer__is_clmbr: bool = getattr(config.data.tokenizer, 'is_clmbr', False)
    tokenizer__is_remap_codes_to_desc: bool = getattr(config.data.tokenizer, 'is_remap_codes_to_desc', False)
    tokenizer__path_to_code_2_detail: str = config.data.tokenizer.path_to_code_2_detail.replace('/local-scratch/nigam/users/hf_ehr/', '/share/pi/nigam/mwornow/hf_ehr/cache/')
    tokenizer__code_2_detail: Dict[str, str] = json.load(open(tokenizer__path_to_code_2_detail, 'r'))

    tokenizer = load_tokenizer_from_config(config)
        
    # Initialize variables for storing patient timelines
    timeline_starts: Dict[int, List[datetime.datetime]] = {}
    timeline_tokens: Dict[int, List[int]] = {}
    timeline_n_dropped_tokens: Dict[int, List[int]] = {} # for tracking the count of dropped tokens at each `timeline_start`
    
    vocab = tokenizer.get_vocab()
    patient_ids, label_values, label_times = [], [], []
    for patient_id, labels in tqdm(labeled_patients.items(), desc=tqdm_desc):
        if allowed_pids is not None and patient_id not in allowed_pids:
            # Skip patients not in `allowed_pids`
            continue
        full_timeline: List[Tuple[datetime.datetime, int]] = [
            (
                e.start, 
                convert_event_to_token(e,
                    tokenizer__code_2_detail, 
                    excluded_vocabs=tokenizer__excluded_vocabs,
                    is_remap_numerical_codes=tokenizer__is_remap_numerical_codes,
                    is_remap_codes_to_desc=tokenizer__is_remap_codes_to_desc,
                    min_code_count=tokenizer__min_code_count,
                    is_clmbr=tokenizer__is_clmbr,
                )
            ) for e in database[patient_id].events
        ]

        # Drop tokens not in vocab
        timeline_with_valid_tokens = [x for x in full_timeline if x[1] in vocab]
        timeline_with_invalid_tokens = [x for x in full_timeline if x[1] not in vocab] # for tracking dropped tokens

        # Tokenize timeline and keep track of each token's start time
        timeline_starts[patient_id] = [x[0] for x in timeline_with_valid_tokens]
        timeline_tokens[patient_id] = tokenizer([x[1] for x in timeline_with_valid_tokens])['input_ids'][0]
        timeline_n_dropped_tokens[patient_id] = [ len([ y for y in timeline_with_invalid_tokens if y[0] <= x[0]]) for x in timeline_with_valid_tokens ]
        assert len(timeline_starts[patient_id]) == len(timeline_tokens[patient_id]), f"Error - timeline_starts and timeline_tokens have different lengths for patient {patient_id}"

        for label in labels:
            patient_ids.append(patient_id)
            label_values.append(label.value)
            label_times.append(label.time)

    return {
        'tokenizer' : tokenizer,
        'patient_ids': patient_ids,
        'label_values': label_values,
        'label_times': label_times,
        'timeline_starts': timeline_starts,
        'timeline_tokens': timeline_tokens,
        'timeline_n_dropped_tokens': timeline_n_dropped_tokens
    }


def main():
    args = parse_args()
    EMBED_STRAT: str = args.embed_strat
    CHUNK_STRAT: str = args.chunk_strat
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    PATH_TO_MODEL = args.path_to_model
    MODEL: str = args.path_to_model.split("/")[-3] if args.model_name in [ None, '' ] else args.model_name
    CKPT: str = get_ckpt_name(PATH_TO_MODEL)
    batch_size: int = args.batch_size if args.batch_size not in [None, ''] else 16
    device: str = args.device
    patient_idx_start: Optional[int] = args.patient_idx_start
    patient_idx_end: Optional[int] = args.patient_idx_end
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, f'{MODEL}_{CKPT}_chunk:{CHUNK_STRAT}_embed:{EMBED_STRAT}')
    os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)

    assert os.path.exists(PATH_TO_MODEL), f"No model exists @ `{PATH_TO_MODEL}`"
    
    logger.critical(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")

    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELED_PATIENTS}`")
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    
    logger.info(f"Loading PatientDatabase from `{PATH_TO_PATIENT_DATABASE}`")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE, read_all=True)
 
    logger.info(f"Loading Config from `{PATH_TO_MODEL}`")
    config = load_config_from_path(PATH_TO_MODEL)

    logger.info(f"Loading Tokenizer from `{PATH_TO_MODEL}")
    tokenizer = load_tokenizer_from_path(PATH_TO_MODEL)

    logger.info(f"Loading Model from `{PATH_TO_MODEL}`")
    model = load_model_from_path(PATH_TO_MODEL)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
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

    # Get patient timelines
    logger.info("Generating patient timelines")
    ehrshot = generate_ehrshot_timelines(database, config, labeled_patients, allowed_pids=allowed_pids)
    patient_ids = ehrshot['patient_ids']
    label_values = ehrshot['label_values']
    label_times = ehrshot['label_times']
    timeline_starts = ehrshot['timeline_starts']
    timeline_tokens = ehrshot['timeline_tokens']
    timeline_n_dropped_tokens = ehrshot['timeline_n_dropped_tokens'] # TODO - track this
    
    # Generate patient representations
    feature_matrix, tokenized_timelines = [], []
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
            
            # Determine how timelines longer than max-length will be chunked
            if CHUNK_STRAT == 'last':
                timelines = [x[-max_length:] for x in timelines]
            else:
                raise ValueError(f"Chunk strategy `{CHUNK_STRAT}` not supported.")

            # Save tokenized version of timelines
            tokenized_timelines.extend([[pad_token_id] * (max_length - len(x)) + x for x in timelines]) # left padding)

            # Create batch
            max_timeline_length = max(len(x) for x in timelines)
            timelines_w_pad = [[pad_token_id] * (max_timeline_length - len(x)) + x for x in timelines] # left padding
            input_ids = torch.stack([torch.tensor(x, device=device) for x in timelines_w_pad])
            attention_mask = (input_ids != pad_token_id).int()
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask if "hyena" not in config['model']['name'] else None
            }
            
            if 'hyena' in config['model']['name']:
                batch.pop('attention_mask')

            # Run model inference
            results = model.model(**batch, output_hidden_states=True)
            hidden_states = results.hidden_states[-1]

            for idx in range(len(pids)):
                if EMBED_STRAT == 'last':
                    patient_rep = hidden_states[idx, -1, :].cpu().numpy()
                elif EMBED_STRAT == 'mean':
                    mask = input_ids[idx] != pad_token_id
                    patient_rep = hidden_states[idx, mask].mean(dim=0).cpu().numpy()
                else:
                    raise ValueError(f"Embedding strategy `{EMBED_STRAT}` not supported.")
                feature_matrix.append(patient_rep)

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
    logger.info(f"Copying model ckpt from `{PATH_TO_MODEL}` to `{path_to_model_ehrshot_dir}`")
    if os.path.exists(path_to_model_ehrshot_dir):
        shutil.rmtree(path_to_model_ehrshot_dir)
    os.makedirs(path_to_model_ehrshot_dir, exist_ok=True)
    shutil.copy(PATH_TO_MODEL, path_to_model_ehrshot_dir)
    ## Copy logging files over
    logger.info(f"Copying logs from `{os.path.join(os.path.dirname(PATH_TO_MODEL), '../logs/')}` to `{path_to_model_ehrshot_dir}/logs`")
    shutil.copytree(os.path.join(os.path.dirname(PATH_TO_MODEL), '../logs/'), os.path.join(path_to_model_ehrshot_dir, 'logs/'))

    # Save EHRSHOT featurization results
    logger.info(f"Stacking featurization results...")
    feature_matrix = np.stack(feature_matrix)
    patient_ids = np.array(patient_ids)
    label_values = np.array(label_values)
    label_times = np.array(label_times)
    tokenized_timelines = np.array(tokenized_timelines)
    assert label_values.shape == label_times.shape, f"Error - label_values and label_times have different shapes: {label_values.shape} vs {label_times.shape}"
    assert label_values.shape == patient_ids.shape, f"Error - label_values and patient_ids have different shapes: {label_values.shape} vs {patient_ids.shape}"
    assert feature_matrix.shape[0] == tokenized_timelines.shape[0], f"Error - feature_matrix and tokenized_timelines have different lengths: {feature_matrix.shape[0]} vs {tokenized_timelines.shape[0]}"
    results = {
        'data_matrix' : feature_matrix, # frozen features from model
        'patient_ids' : patient_ids,
        'labeling_time' : label_times,
        'label_values' : label_values,
        'tokenized_timelines' : tokenized_timelines, # tokenized timeline for each patient in EHRSHOT
        'wandb_run_id' : wandb_run_id,
        'path_to_ckpt_ehrshot' : path_to_model_ehrshot_dir,
        'path_to_ckpt_orig' : PATH_TO_MODEL,
    }

    path_to_features_pkl: str = PATH_TO_OUTPUT_FILE + (f'--start_idx={patient_idx_start}' if patient_idx_start else '') + (f'--end_idx={patient_idx_end}' if patient_idx_end else '') + '_features.pkl'
    logger.info(f"Saving results to `{path_to_features_pkl}`")
    with open(path_to_features_pkl, 'wb') as f:
        pickle.dump(results, f)

    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids[:10])}\n"
                f"label_values={repr(label_values[:10])}\n"
                f"label_times={repr(label_times[:10])}\n"
                f"tokenized_timelines={tokenized_timelines[0][-10:]}")
    logger.info(f"Shapes: feature_matrix={feature_matrix.shape}, patient_ids={patient_ids.shape}, label_values={label_values.shape}, label_times={label_times.shape}, tokenized_timelines={tokenized_timelines.shape}")
    logger.success("Done!")

if __name__ == "__main__":
    main()
