import argparse
import datetime
import os
import pickle
import json
from omegaconf import OmegaConf

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
    --path_to_ehrshot_dir /share/pi/nigam/mwornow/ehrshot-benchmark/ \
    --path_to_model /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models \
    --path_to_tokenizer /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json \
    --embed_strat last \
    --chunk_strat last
'''

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patient representations (for all tasks at once)")
    parser.add_argument("--path_to_ehrshot_dir", required=True, type=str, help="Path to EHRSHOT directory")
    parser.add_argument("--path_to_model", type=str, help="Path to model checkpoint. If this is a directory, will load `last.ckpt`. If this is a file ending in `.ckpt`, it will load that file directly. ")
    parser.add_argument("--path_to_tokenizer", default='/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json', type=str, help="Path to tokenizer. Defaults to tokenizer_v9_lite")
    parser.add_argument("--embed_strat", type=str, help="Strategy used for condensing a chunk of a timeline into a single embedding. Options: 'last' (only take last token), 'avg' (avg all tokens).")
    parser.add_argument("--chunk_strat", type=str, help="Strategy used for condensing a timeline longer than context window C. Options: 'last' (only take last chunk), 'avg' (avg all chunks together).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    embed_strat: str = args.embed_strat
    chunk_strat: str = args.chunk_strat
    path_to_model = os.path.join(args.path_to_model, MODEL, 'epoch=2-step=56000-val_loss.ckpt')
    path_to_tokenizer: str = args.path_to_tokenizer
    path_to_ehrshot_dir: str = args.path_to_ehrshot_dir
    path_to_dataset: str = os.path.join(path_to_ehrshot_dir, 'ehrshot-meds-stanford', 'data', 'data.parquet')
    path_to_labels_csv: str = os.path.join(path_to_ehrshot_dir, 'assets', 'labels',  'merged_labels.csv')
    path_to_output_file: str = os.path.join(path_to_ehrshot_dir, 'assets', 'features', f'{MODEL}_chunk:{chunk_strat}_embed:{embed_strat}_features.pkl')
    
    # Check that requested model exists
    assert os.path.exists(path_to_model), f"No such path exists @ `{path_to_model}`"
    assert os.path.exists(path_to_labels_csv), f"No labels CSV found at `{path_to_labels_csv}`"
    assert os.path.exists(path_to_dataset), f"No EHRSHOT dataset found at `{path_to_dataset}`"
    assert os.path.exists(os.path.dirname(path_to_output_file)), f"No EHRSHOT `features/` folder found at `{path_to_output_file}`"

    # Load model config
    # Directory structure is: ..../model_name/ckpts/last.ckpt, and use can provide any one of these
    if os.path.isdir(path_to_model) and path_to_model.split('/')[-1] != 'ckpts':
        # We're at .../model_name/...
        path_to_config: str = os.path.join(path_to_model, '.hydra', 'config.yaml')
        path_to_ckpt: str = os.path.join(path_to_model, 'ckpts', 'last.ckpt')
    elif os.path.isdir(path_to_model) and path_to_model.split('/')[-1] == 'ckpts':
        # We're at .../ckpts/...
        path_to_config: str = os.path.join(path_to_model, '../', '.hydra', 'config.yaml')
        path_to_ckpt: str = os.path.join(path_to_model, 'last.ckpt')
    elif not os.path.isdir(path_to_model) and path_to_model.endswith('.ckpt'):
        path_to_config: str = os.path.join(path_to_model, '../', '.hydra', 'config.yaml')
        path_to_ckpt: str = path_to_model
    else:
        raise ValueError(f"Couldn't figure out how to load model from `{path_to_model}`")
    assert os.path.exists(path_to_config), f"No config.yaml exists @ `{path_to_config}`"
    assert os.path.exists(path_to_model), f"No model exists @ `{path_to_model}`"

    # Load EHRSHOT dataset
    dataset = datasets.Dataset.from_parquet(path_to_dataset)
    
    # Load consolidated labels across all patients for all tasks
    labels: List[meds.Label] = convert_csv_labels_to_meds(path_to_labels_csv)
 
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise ValueError('No cuda device found. Throwing error')

    # Tokenizer
    logger.info(f"Loading tokenizer: `{path_to_tokenizer}`")
    femr_vocab_count: Dict[str, int] = json.load(open(path_to_tokenizer, 'r'))
    tokenizer = FEMRTokenizer(femr_vocab_count, min_code_count=tokenizer_min_code_count)
    logger.info(f"Vocab size: `{tokenizer.vocab_size}`")
    
    # Model
    logger.info(f"Loading model: `{path_to_model}`")
    config: OmegaConf = OmegaConf.load(path_to_config)
    checkpoint = torch.load(path_to_model)
    if 'gpt2' in config.model.name:
        model = GPTLanguageModel(**checkpoint['hyper_parameters'])
    elif 'bert' in config.model.name:
        model = BERTLanguageModel(**checkpoint['hyper_parameters'])
    elif 'hyena' in config.model.name:
        model = HyenaLanguageModel(**checkpoint['hyper_parameters'])
    elif 'mamba' in config.model.name:
        model = MambaLanguageModel(**checkpoint['hyper_parameters'])
    elif 't5' in config.model.name:
        model = T5LanguageModel(**checkpoint['hyper_parameters'])
    else:
        raise ValueError(f"Model `{config.model.name}` not supported.")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    logger.info(f"Parameter count of model = {model.get_param_count()}")

    # Generate patient representations
    max_length: int = model.config.data.dataloader.max_length
    patient_ids, feature_times, features = [], [], []
    with torch.no_grad():
        for label in tqdm(labels, desc='Generating patient representations', total=len(labels)):            
            patient_id: int = label['patient_id']
            prediction_time = label['prediction_time']

            # Truncate timeline of codes to only events that start <= label.time
            timeline: List[str] = []
            for event in dataset[patient_id]['events']:
                if event['time'] <= prediction_time:
                    for m in event['measurements']:
                        timeline.append(m['code'])
                else:
                    break

            # Tokenize
            tokens: List[int] = tokenizer.encode(timeline, truncate=False)
            
            # Chunking
            if chunk_strat == 'last':
                tokens = tokens[-max_length:]
            else:
                raise ValueError(f"Chunk strategy `{chunk_strat}` not supported.")

            # Inference
            # logits.shape = (batch_size = 1, sequence_length, vocab_size = 167k)
            # hidden_states.shape = (batch_size = 1, sequence_length, hidden_size = 768)
            logits, hidden_states = model({ 'input_ids': torch.tensor([ tokens ]).to(device) })

            # Aggregate embeddings
            if embed_strat == 'last':
                patient_rep = hidden_states[:,-1,:].detach().cpu().numpy()
            elif embed_strat == 'avg':
                patient_rep = hidden_states.mean(dim=1).detach().cpu().numpy()
            else:
                raise ValueError(f"Embedding strategy `{embed_strat}` not supported.")
            
            # Save results
            patient_ids.append(patient_id)
            features.append(patient_rep)
            feature_times.append(prediction_time)

    results: Dict[str, Any] = {
        'patient_ids' : np.array(patient_ids),
        'feature_times' : np.array(feature_times),
        'features' : np.concatenate(features),
    }

    # Save results
    path_to_output_file = os.path.join(path_to_features_dir, f"{MODEL_NAME}_features.pkl")
    logger.info(f"Saving results to `{path_to_output_file}`")
    with open(path_to_output_file, 'wb') as f:
        pickle.dump(results, f)

    # Logging
    patient_ids, feature_times, features = results['patient_ids'], results['feature_times'], results['features']
    logger.info("FeaturizedPatient stats:\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"features={repr(features)}\n"
                f"feature_times={repr(feature_times)}\n")
    logger.success("Done!")