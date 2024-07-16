import argparse
import datetime
import os
import json
import torch
import femr.datasets
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from typing import Dict, Any, List, Optional, Tuple, Union
from omegaconf import DictConfig
from jaxtyping import Float

from femr.labelers import LabeledPatients, load_labeled_patients
from hf_ehr.data.tokenization import FEMRTokenizer, DescTokenizer
from hf_ehr.data.datasets import convert_event_to_token
from hf_ehr.models.bert import BERTLanguageModel
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.models.hyena import HyenaLanguageModel
from hf_ehr.models.t5 import T5LanguageModel
from hf_ehr.models.mamba import MambaLanguageModel
from hf_ehr.models.modules import BaseModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a model for EHRSHOT tasks")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_ehrshot_split", required=True, type=str, help="Path to EHRSHOT split CSV")
    parser.add_argument("--path_to_model", type=str, required=True, help="Path to model .ckpt")
    parser.add_argument("--finetune_type", type=str, required=True, choices=["full", "last_n_layers"], help="Type of finetuning")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers to finetune if finetune_type is 'last_n_layers'")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned models")
    parser.add_argument("--embed_strat", type=str, help="Strategy used for condensing a chunk of a timeline into a single embedding. Options: 'last' (only take last token), 'avg' (avg all tokens).")
    parser.add_argument("--chunk_strat", type=str, help="Strategy used for condensing a timeline longer than context window C. Options: 'last' (only take last chunk), 'avg' (avg all chunks together).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run training on")
    return parser.parse_args()

def get_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    config = checkpoint['hyper_parameters']['config']
    def recurse(d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if v == 'None':
                d[k] = None
            elif isinstance(v, dict):
                recurse(v)
    recurse(config)
    return config

def load_model(path_to_model: str, tokenizer: Any) -> nn.Module:
    checkpoint = torch.load(path_to_model, map_location='cpu')
    model_map = {
        'bert': BERTLanguageModel,
        'gpt': GPTLanguageModel,
        'hyena': HyenaLanguageModel,
        'mamba': MambaLanguageModel,
        't5': T5LanguageModel
    }
    MODEL = path_to_model.split("/")[-3]
    model_class = next((m for k, m in model_map.items() if k in MODEL), None)
    if not model_class:
        raise ValueError(f"Model `{MODEL}` not supported.")
    model = model_class(**checkpoint['hyper_parameters'], tokenizer=tokenizer)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def add_linear_head(model: nn.Module, device: str) -> nn.Module:
    hidden_size = model.config.model.config_kwargs.get('d_model', None)
    if hidden_size is None:
        hidden_size = model.config.model.config_kwargs.get('n_embd', None)
    
    if hidden_size is None:
        raise ValueError("Neither 'd_model' nor 'n_embd' is defined in the model's config")

    model.classifier = nn.Linear(hidden_size, 2)  # Assuming binary classification
    model.classifier = model.classifier.to(device)
    return model

def setup_finetuning(model: nn.Module, finetune_type: str, n_layers: int) -> nn.Module:
    if finetune_type == "last_n_layers":
        for param in model.parameters():
            param.requires_grad = False
        for param in list(model.parameters())[-n_layers:]:
            param.requires_grad = True
    elif finetune_type == "full":
        for param in model.parameters():
            param.requires_grad = True
    return model

def save_finetuned_model(model: nn.Module, ckpt_name: str, ehrshot_task_name: str, fine_tune_strat: str, output_dir: str) -> None:
    output_path = os.path.join(output_dir, f"ckpt={ckpt_name}--ehrshot={ehrshot_task_name}--finetune={fine_tune_strat}.pt")
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load EHRSHOT split CSV
    ehrshot_split_df = pd.read_csv(args.path_to_ehrshot_split)
    train_patient_ids = set(ehrshot_split_df[ehrshot_split_df['split'] == 'train']['omop_person_id'])

    labeled_patients: LabeledPatients = load_labeled_patients(os.path.join(args.path_to_labels_dir, 'all_labels.csv'))
    database = femr.datasets.PatientDatabase(args.path_to_database, read_all=True)

    checkpoint = torch.load(args.path_to_model, map_location='cpu')
    config = get_config(checkpoint)

    tokenizer__path_to_code_2_detail: str = config.data.tokenizer.path_to_code_2_detail.replace('/local-scratch/nigam/users/hf_ehr/', '/share/pi/nigam/mwornow/hf_ehr/cache/')
    tokenizer__excluded_vocabs: Optional[List[str]] = config.data.tokenizer.excluded_vocabs
    tokenizer__min_code_count: Optional[int] = config.data.tokenizer.min_code_count if hasattr(config.data.tokenizer, 'min_code_count') else None
    tokenizer__is_remap_numerical_codes: bool = config.data.tokenizer.is_remap_numerical_codes if hasattr(config.data.tokenizer, 'is_remap_numerical_codes') else False
    tokenizer__is_clmbr: bool = config.data.tokenizer.is_clmbr if hasattr(config.data.tokenizer, 'is_clmbr') else False
    tokenizer__is_remap_codes_to_desc: bool = config.data.tokenizer.is_remap_codes_to_desc if hasattr(config.data.tokenizer, 'is_remap_codes_to_desc') else False
    tokenizer__desc_emb_tokenizer: bool = config.data.tokenizer.desc_emb_tokenizer if hasattr(config.data.tokenizer, 'desc_emb_tokenizer') else False
    tokenizer__code_2_detail: Dict[str, str] = json.load(open(tokenizer__path_to_code_2_detail, 'r'))

    if tokenizer__is_clmbr:
        tokenizer = FEMRTokenizer(tokenizer__path_to_code_2_detail, 
                                  excluded_vocabs=tokenizer__excluded_vocabs,
                                  is_remap_numerical_codes=tokenizer__is_remap_numerical_codes,
                                  min_code_count=tokenizer__min_code_count)
    elif tokenizer__is_remap_codes_to_desc:
        tokenizer = DescTokenizer(AutoTokenizer.from_pretrained(tokenizer__desc_emb_tokenizer))
    else:
        tokenizer = FEMRTokenizer(tokenizer__path_to_code_2_detail, 
                                    excluded_vocabs=tokenizer__excluded_vocabs,
                                    is_remap_numerical_codes=tokenizer__is_remap_numerical_codes,
                                    min_code_count=tokenizer__min_code_count)

    model = load_model(args.path_to_model, tokenizer)
    model.to(args.device)
    model = setup_finetuning(model, args.finetune_type, args.n_layers)
    model = add_linear_head(model, args.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()  # Set the model to training mode

    max_length: int = config.data.dataloader.max_length
    pad_token_id: int = tokenizer.token_2_idx['[PAD]']
    feature_matrix, patient_ids, label_values, label_times = [], [], [], []
    
        # Iterate over each subdirectory in the labels directory
    for ehrshot_task in os.listdir(args.path_to_labels_dir):
        task_labels_dir = os.path.join(args.path_to_labels_dir, ehrshot_task)
        if not os.path.isdir(task_labels_dir):
            continue

        labeled_patients: LabeledPatients = load_labeled_patients(os.path.join(task_labels_dir, 'labeled_patients.csv'))

        pat_num = 0
        for patient_id, labels in tqdm(labeled_patients.items(), desc=f"Loading EHRSHOT patient timelines for task {ehrshot_task}"):
            if patient_id not in train_patient_ids:
                continue  # Skip patients not in the 'train' split
            if pat_num > 10:
                break
            pat_num += 1
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

            timeline_with_valid_tokens = [x for x in full_timeline if x[1] in tokenizer.get_vocab()]
            timeline_starts = [x[0] for x in timeline_with_valid_tokens]
            timeline_tokens = tokenizer([x[1] for x in timeline_with_valid_tokens])['input_ids'][0]
            assert len(timeline_starts) == len(timeline_tokens), f"Error - timeline_starts and timeline_tokens have different lengths for patient {patient_id}"

            for label in labels:
                patient_ids.append(patient_id)
                label_values.append(label.value)
                label_times.append(label.time)

        for batch_start in tqdm(range(0, len(patient_ids), args.batch_size), desc='Generating patient representations', total=len(patient_ids) // args.batch_size):
            pids = patient_ids[batch_start:batch_start + args.batch_size]
            values = label_values[batch_start:batch_start + args.batch_size]
            times = label_times[batch_start:batch_start + args.batch_size]
            timelines = []

            for pid, l_value, l_time in zip(pids, values, times):
                timeline_starts_for_pid = timeline_starts
                timeline_tokens_for_pid = timeline_tokens
                timeline = [token for start, token in zip(timeline_starts_for_pid, timeline_tokens_for_pid) if start <= l_time]
                timelines.append(timeline)

            if args.chunk_strat == 'last':
                timelines = [x[-max_length:] for x in timelines]
            else:
                raise ValueError(f"Chunk strategy `{args.chunk_strat}` not supported.")

            max_timeline_length = max(len(x) for x in timelines)
            timelines_w_pad = [[pad_token_id] * (max_timeline_length - len(x)) + x for x in timelines]
            # Check lengths and content of timelines_w_pad
            filtered_timelines = []
            for idx, t in enumerate(timelines_w_pad):
                if len(t) != max_timeline_length:
                    print(f"Length mismatch at index {idx}: {len(t)} != {max_timeline_length}")
                    print(f"Content: {t}")
                    continue
                
                if all(token == pad_token_id for token in t):
                    print(f"Skipping patient at index {idx} with only padding tokens.")
                    continue

                filtered_timelines.append(t)

            if not filtered_timelines:
                print("All patient timelines were skipped due to padding. Exiting...")
                continue

            try:
                input_ids = torch.stack([torch.tensor(x, device=args.device) for x in filtered_timelines])
            except Exception as e:
                # Debugging statements to inspect filtered_timelines when an error occurs
                print("Error occurred during torch.stack. filtered_timelines:")
                for t in filtered_timelines:
                    print(t)
                raise e
            #input_ids = torch.stack([torch.tensor(x, device=args.device) for x in timelines_w_pad])
            attention_mask = (input_ids != pad_token_id).int().to(args.device)
            
            # Shift labels by one token for language modeling
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = pad_token_id
            labels = labels.long().to(args.device)

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

            optimizer.zero_grad()
            results = model.model(**batch, output_hidden_states=True)
            
            # Ensure hidden_states are on the same device as the classifier
            hidden_states = results.hidden_states[-1]
            hidden_states = hidden_states.to(args.device)

            # Compute classification logits
            classification_logits = model.classifier(hidden_states[:, 0, :])
            
            # Compute classification loss using binary labels
            binary_labels = torch.tensor(values, device=args.device).long()
            loss = criterion(classification_logits, binary_labels)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

    
        # Save the fine-tuned model
        save_finetuned_model(model, ckpt_name=os.path.basename(args.path_to_model).split('.')[0], ehrshot_task_name=ehrshot_task, fine_tune_strat=args.finetune_type, output_dir=args.output_dir)

    logger.success("Done!")

if __name__ == "__main__":
    main()
