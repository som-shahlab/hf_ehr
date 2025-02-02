"""
Usage:
    python hf_upload.py

Purpose:
    Upload an hf_ehr model+tokenizer to Hugging Face Hub.
"""

import json
import os
import torch
import yaml
import shutil
from tqdm import tqdm
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config, LlamaForCausalLM, LlamaConfig, MambaForCausalLM, MambaConfig, AutoConfig, AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo, upload_folder
from safetensors.torch import save_model
from hf_ehr.scripts.huggingface.hf_readme import readme_text

def get_param_count(model) -> int:
    """Returns the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

HF_USERNAME = "Miking98"
HF_TOKEN = os.getenv("HF_TOKEN")

base_dir: str = '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/'
models = [ 
    'gpt-base-512--clmbr',
    'gpt-base-1024--clmbr',
    'gpt-base-2048--clmbr',
    'gpt-base-4096--clmbr',
    'hyena-large-1024--clmbr',
    'hyena-large-4096--clmbr',
    'hyena-large-8192--clmbr',
    'hyena-large-16384--clmbr',
    'mamba-tiny-1024--clmbr',
    'mamba-tiny-4096--clmbr',
    'mamba-tiny-8192--clmbr',
    'mamba-tiny-16384--clmbr', 
    'llama-base-512--clmbr',
    'llama-base-1024--clmbr',
    'llama-base-2048--clmbr',
    'llama-base-4096--clmbr',
]

for model_name in tqdm(models):
    path_to_model: str = os.path.join(base_dir, model_name)
    path_to_ckpt: str = os.path.join(path_to_model, 'ckpts', 'train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt')
    path_to_config: str = os.path.join(path_to_model, 'logs', 'artifacts', 'config.yaml')
    path_to_tokenizer: str = os.path.join(path_to_model, 'logs', 'artifacts', 'tokenizer_config.json')

    # Load checkpoint
    ckpt = torch.load(path_to_ckpt, map_location='cpu')

    # Load config
    with open(path_to_config) as f:
        config = yaml.safe_load(f)
    print("Training config:", config['model']['config_kwargs'])
    
    # Force some configs
    config['model']['config_kwargs']['vocab_size'] = 39818
    config['model']['config_kwargs']['bos_token_id'] = 0
    config['model']['config_kwargs']['eos_token_id'] = 1
    config['model']['config_kwargs']['unk_token_id'] = 2
    config['model']['config_kwargs']['sep_token_id'] = 3
    config['model']['config_kwargs']['pad_token_id'] = 4
    config['model']['config_kwargs']['cls_token_id'] = 5
    config['model']['config_kwargs']['mask_token_id'] = 6
    
    # Model-specific configs
    if 'gpt' in model_name:
        config['model']['config_kwargs']['n_ctx'] = config['model']['config_kwargs']['n_positions']
    elif 'llama' in model_name:
        config['model']['config_kwargs']['rope_scaling'] = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "rope_type": "llama3",
            # NOTE: Setting original == max will cause this warning to throw: https://github.com/huggingface/transformers/blob/62db3e6ed67a74cc1ed1436acd9973915c0a4475/src/transformers/modeling_rope_utils.py#L534-L538
            # But we can ignore this b/c it will just cause the attention_factor to be set to 1.0
            "original_max_position_embeddings": config['model']['config_kwargs']['max_position_embeddings'],
        }
        config['model']['config_kwargs']['head_dim'] = config['model']['config_kwargs']['hidden_size'] // config['model']['config_kwargs']['num_attention_heads']
    elif 'hyena' in model_name:
        # NOTE: By default, Hyena sets `pad_vocab_size_multiple` to 8
        # This increases our vocab size from 39818 => 39824, which makes `lm_head.weight.shape = (39824, 768)`
        # See: https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen-hf/blob/42dedd4d374eac0fb8168549e546a3472fbd27ae/configuration_hyena.py#L27
        # Here, we undo this.
        config['model']['config_kwargs']['pad_vocab_size_multiple'] = 1

    # Instantiate model and load weights
    new_state_dict = ckpt['state_dict']
    if 'gpt' in model_name:
        new_state_dict = {k.replace('model.', ''): v for k, v in new_state_dict.items()}
    elif 'hyena' in model_name:
        new_state_dict = {k.replace('model.', ''): v for k, v in new_state_dict.items()}
        # NOTE: By default, Hyena sets `pad_vocab_size_multiple` to 8
        # This increases our vocab size from 39818 => 39824, which makes `lm_head.weight.shape = (39824, 768)`
        # See: https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen-hf/blob/42dedd4d374eac0fb8168549e546a3472fbd27ae/configuration_hyena.py#L27
        # Here, we undo this.
        vocab_size: int = config['model']['config_kwargs']['vocab_size']
        new_state_dict['lm_head.weight'] = new_state_dict['lm_head.weight'][:vocab_size]
        new_state_dict['hyena.backbone.embeddings.word_embeddings.weight'] = new_state_dict['hyena.backbone.embeddings.word_embeddings.weight'][:vocab_size]
    elif 'mamba' in model_name:
        new_state_dict = {k.replace('model.', ''): v for k, v in new_state_dict.items()}
    elif 'llama' in model_name:
        new_state_dict = {k.replace('model.model.', 'model.'): v for k, v in new_state_dict.items()}
        new_state_dict = {k.replace('model.lm_head.', 'lm_head.'): v for k, v in new_state_dict.items()}
    elif 'based' in model_name:
        new_state_dict = {k.replace('model.', ''): v for k, v in new_state_dict.items()}
    else:
        raise ValueError(f"Model `{model_name}` not supported.")
    
    # Create HuggingFace config
    hf_config = AutoConfig.from_pretrained(config['model']['hf_name'], trust_remote_code=True)
    for key, val in config['model']['config_kwargs'].items():
        setattr(hf_config, key, val)
    print("\nHuggingFace config:", hf_config)

    # Create HuggingFace model
    model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    model.load_state_dict(new_state_dict, strict=True)
    
    # Remap "--" => "-" b/c HF doesn't support "--" in repo names
    model_name = model_name.replace('--', '-')

    # Save model + tokenizer to local directory
    path_to_local_dir = "../../../cache/hf_model_for_upload"
    os.makedirs(path_to_local_dir, exist_ok=True)
    ## Model
    model.save_pretrained(path_to_local_dir, safe_serialization=False)
    save_model(model, os.path.join(path_to_local_dir, "model.safetensors"), metadata={'format': 'pt'})

    ## Tokenizer
    shutil.copy(path_to_tokenizer, os.path.join(path_to_local_dir, 'tokenizer_config.json'))
    with open(os.path.join(path_to_local_dir, 'tokenizer_config.json'), 'r') as f:
        tokenizer_config = json.load(f)
    for i in range(len( tokenizer_config['tokens'])):
        tokenizer_config['tokens'][i]['stats'] = [] # drop stats since they're null
    with open(os.path.join(path_to_local_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    ## README
    base_model: str = model_name.split('-')[0]
    ctx_length: int = model_name.split('-')[2]
    param_count: int = get_param_count(model)
    with open(os.path.join(path_to_local_dir, "README.md"), "w") as f:
        f.write(readme_text(model_name, base_model, ctx_length, param_count))

    # Push model to Hugging Face Hub
    api = HfApi(token=HF_TOKEN)
    repo_id = f"StanfordShahLab/{model_name}"
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder( folder_path=path_to_local_dir, repo_id=repo_id, delete_patterns=['.bin', '.safetensors'])

    print("Model and tokenizer successfully pushed to the Hugging Face Hub!")