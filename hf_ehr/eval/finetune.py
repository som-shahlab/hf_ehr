import argparse
import json
import os
import torch
import femr.datasets
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from loguru import logger
from jaxtyping import Float

from femr.labelers import LabeledPatients, load_labeled_patients
from hf_ehr.utils import load_config_from_path, load_tokenizer_from_path, load_model_from_path
from hf_ehr.eval.ehrshot import generate_ehrshot_timelines

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a model for EHRSHOT tasks")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_ehrshot_split", required=True, type=str, help="Path to EHRSHOT split CSV")
    parser.add_argument("--path_to_model", type=str, required=True, help="Path to model .ckpt")
    parser.add_argument("--finetune_strat", type=str, required=True, choices=["full", "last_n_layers"], help="Type of finetuning")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers to finetune if finetune_strat is 'last_n_layers'")
    parser.add_argument("--path_to_output_dir", type=str, required=True, help="Directory to save fine-tuned models")
    parser.add_argument("--embed_strat", type=str, help="Strategy used for condensing a chunk of a timeline into a single embedding. Options: 'last' (only take last token), 'avg' (avg all tokens).")
    parser.add_argument("--chunk_strat", type=str, help="Strategy used for condensing a timeline longer than context window C. Options: 'last' (only take last chunk), 'avg' (avg all chunks together).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for Adam")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run training on")
    return parser.parse_args()

class CookbookModelWithClassificationHead(nn.Module):
    def __init__(self, model: nn.Module, aggregation_strat: str, n_classes: int):
        super().__init__()
        self.n_classes: int = n_classes
        self.hidden_dim: int = model.model.lm_head.in_features
        # Base model
        if model.model.__class__.__name__ == 'MambaForCausalLM':
            self.base_model = model.model.backbone
            self.base_model_name = 'mamba'
        else:
            raise ValueError("Model must be a MambaForCausalLM")
        # Aggregation of base model reprs for classification
        if aggregation_strat == 'mean':
            self.aggregation = lambda x: torch.mean(x, dim=1)
        elif aggregation_strat == 'max':
            self.aggregation = lambda x: torch.max(x, dim=1)
        elif aggregation_strat == 'last':
            self.aggregation = lambda x: x[:, -1, :]
        elif aggregation_strat == 'first':
            self.aggregation = lambda x: x[:, 0, :]
        else:
           raise ValueError(f"Aggregation strategy `{aggregation_strat}` not supported.") 
        # Linear head
        self.classifier = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, input_ids: Float[torch.Tensor, 'B L'] = None, attention_mask: Float[torch.Tensor, 'B L'] = None, **kwargs) -> Float[torch.Tensor, 'B C']:
        """Return logits for classification task"""
        reprs: Float[torch.Tensor, 'B L H'] = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        agg: Float[torch.Tensor, 'B H'] = self.aggregation(reprs)
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

def setup_finetuning(model: CookbookModelWithClassificationHead, finetune_strat: str, n_layers: int):
    # Start by freezing all `base_model` params
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Find layers we can unfreeze
    if model.base_model_name == 'mamba':
        layers = model.base_model.layers
    elif model.base_model_name == 'hyena':
        assert False
    elif model.base_model_name == 'gpt2':
        assert False
    elif model.base_model_name == 'bert':
        assert False
    else:
        raise ValueError(f"Base model `{model.base_model_name}` not supported.")

    # Selectively unfreeze, depending on `finetune_strat`
    if finetune_strat == "last_n_layers":
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif finetune_strat == "full":
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    elif finetune_strat == 'frozen':
        pass
    else:
        raise ValueError(f"Fine-tuning strategy `{finetune_strat}` not supported.")
    return model

def save_finetuned_model(model: nn.Module, 
                         ckpt_name: str, 
                         ehrshot_task_name: str, 
                         fine_tune_strat: str, 
                         embed_strat: str,
                         chunk_strat: str,
                         n_layers: int,
                         path_to_output_dir: str) -> None:
    path_to_output: str = os.path.join(path_to_output_dir, f"ckpt={ckpt_name}--ehrshot_task={ehrshot_task_name}--finetune_strat={fine_tune_strat}--n_layers={n_layers}--embed_strat={embed_strat}--chunk_strat={chunk_strat}.pt")
    torch.save(model.state_dict(), path_to_output)
    logger.info(f"Model saved to {path_to_output}")

def main() -> None:
    args = parse_args()
    batch_size: int = args.batch_size
    chunk_strat: str = args.chunk_strat
    path_to_output_dir: str = args.path_to_output_dir if args.path_to_output_dir is not None else os.path.join(os.path.dirname(args.path_to_model), './finetunes')
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Load EHRSHOT split CSV
    logger.info(f"Loading EHRSHOT split from `{args.path_to_ehrshot_split}`")
    df_ehrshot_split = pd.read_csv(args.path_to_ehrshot_split)
    train_patient_ids = set(df_ehrshot_split[df_ehrshot_split['split'] == 'train']['omop_person_id'])

    logger.info(f"Loading PatientDatabase from `{args.path_to_database}`")
    database = femr.datasets.PatientDatabase(args.path_to_database, read_all=True)

    logger.info(f"Loading Tokenizer from `{args.path_to_model}")
    tokenizer = load_tokenizer_from_path(args.path_to_model)
 
    logger.info(f"Loading Config from `{args.path_to_model}`")
    config = load_config_from_path(args.path_to_model)

    logger.info(f"Loading Model from `{args.path_to_model}`")
    model = load_model_from_path(args.path_to_model, tokenizer)
    model = CookbookModelWithClassificationHead(model, args.embed_strat, 2)
    model = setup_finetuning(model, args.finetune_strat, args.n_layers)
    model.to(args.device)
    model.train()
    
    # Optimizer + Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.functional.cross_entropy

    # Iterate over each subdirectory in the `labels/` directory
    for task in os.listdir(args.path_to_labels_dir):
        path_to_task_dir: str = os.path.join(args.path_to_labels_dir, task)
        if not os.path.isdir(path_to_task_dir):
            continue

        # Load labels + few shots
        path_to_few_shots: str = os.path.join(path_to_task_dir, 'all_shots_data.json')
        few_shots: dict = json.load(open(path_to_few_shots, 'r'))

        for subtask in few_shots.keys():
            for shot in few_shots[subtask].keys():
                for replicate in few_shots[subtask][shot].keys():
                    data = few_shots[subtask][shot][replicate]

                    # Get labels + patient timelines
                    labeled_patients: LabeledPatients = load_labeled_patients(os.path.join(path_to_task_dir, 'labeled_patients.csv'))
                    ehrshot = generate_ehrshot_timelines(database, 
                                                            config, 
                                                            labeled_patients, 
                                                            allowed_pids=train_patient_ids, 
                                                            tqdm_desc=f"Loading EHRSHOT patient timelines for task={task} | subtask={subtask} | shot={shot} | replicate={replicate}")
                    patient_ids = ehrshot['patient_ids']
                    label_values = ehrshot['label_values']
                    label_times = ehrshot['label_times']
                    timeline_starts = ehrshot['timeline_starts']
                    timeline_tokens = ehrshot['timeline_tokens']

                    # Generate patient representations
                    max_length: int = config.data.dataloader.max_length
                    pad_token_id: int = tokenizer.token_2_idx['[PAD]']
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
                        if chunk_strat == 'last':
                            timelines = [x[-max_length:] for x in timelines]
                        else:
                            raise ValueError(f"Chunk strategy `{chunk_strat}` not supported.")

                        # Create batch
                        max_timeline_length = max(len(x) for x in timelines)
                        timelines_w_pad = [[pad_token_id] * (max_timeline_length - len(x)) + x for x in timelines] # left padding
                        input_ids = torch.stack([torch.tensor(x, device=args.device) for x in timelines_w_pad])
                        attention_mask = (input_ids != pad_token_id).int()
                        batch = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask if "hyena" not in config['model']['name'] else None,
                        }
                        
                        # Check lengths and content of timelines_w_pad
                        for idx, t in enumerate(timelines_w_pad):
                            assert len(t) == max_timeline_length, f"Length mismatch at index {idx}: {len(t)} != {max_timeline_length} | Content: {t}"
                            assert not all(token == pad_token_id for token in t), f"Found patient at index {idx} with only padding tokens."

                        # Run model to get logits for each class
                        logits: Float[torch.Tensor, 'B C'] = model(**batch)

                        # Compute CE loss v. true binary labels
                        binary_labels: Float[torch.Tensor, 'B'] = torch.tensor(values, device=args.device).long()
                        loss = criterion(logits, binary_labels)

                        # Backprop
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                    # Save the fine-tuned model
                    save_finetuned_model(model, 
                                        ckpt_name=os.path.basename(args.path_to_model).split('.')[0], 
                                        ehrshot_task_name=ehrshot_task, 
                                        fine_tune_strat=args.finetune_strat, 
                                        embed_strat=args.embed_strat,
                                        chunk_strat=args.chunk_strat,
                                        n_layers=args.n_layers,
                                        path_to_output_dir=path_to_output_dir)

    logger.success("Done!")

if __name__ == "__main__":
    main()
