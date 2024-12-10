import json
import os
import torch
import yaml
import shutil
from tqdm import tqdm
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config, LlamaForCausalLM, LlamaConfig, MambaForCausalLM, MambaConfig, AutoConfig, AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo, upload_folder

def get_param_count(model) -> int:
    """Returns the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

readme_text = lambda model_name, base_model, ctx_length, param_count: f"""
---
license: cc-by-nc-4.0
library_name: {model_name}
tags:
- healthcare
- medical
extra_gated_prompt: "You agree to all terms outlined in 'The EHRSHOT Credentialed Health Data License' (see https://shahlab.stanford.edu/ehrshot_license). Access requires a verified CITI training certificate using the same process outlined by PhysioNet (see https://physionet.org/about/citi-course/). Please complete the 'Data or Specimens Only Research' course and please provide proof via the verification URL, which takes the form https://www.citiprogram.org/verify/?XXXXXX. You agree to not use the model to conduct experiments that cause harm to human subjects." 
extra_gated_fields:
 Full Name: text
 Email: text
 Affiliation: text
 CITI Certification Verification URL: text
 I agree to all terms outlined in 'The EHRSHOT Credentialed Health Data License': checkbox
 I agree to use this model for non-commercial use ONLY: checkbox
---

# {model_name}

This is a **{base_model}** model with context length **{ctx_length}** with **{param_count}** parameters from the [Context Clues paper](TODO).

It is a foundation model trained from scratch on the structured data within 2.57 million deidentified EHRs from Stanford Medicine.

As input, this model expects a sequence of coded medical events that have been mapped to Standard Concepts within the [OMOP-CDM vocabulary](https://ohdsi.github.io/CommonDataModel/index.html). As output, the model can generate either (a) synthetic future timelines or (b) a vector representation of a patient which can then be used for downstream prediction tasks.

## Usage

First, install the `hf-ehr` package:
```bash
pip install transformers torch hf-ehr
```

Second, run this Python script to do inference on a patient representation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from typing import List, Dict
import torch

####################################
# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("StanfordShahLab/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("StanfordShahLab/{model_name}")

####################################
# 2. Define patient as sequence of `Event` objects. Only `code` is required.
patient: List[Event] = [
    Event(code='SNOMED/3950001', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='Gender/F', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='Ethnicity/Hispanic', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='SNOMED/609040007', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='LOINC/2236-8', value=-3.0, unit=None, start=None, end=None, omop_table=None),
    Event(code='SNOMED/12199005', value=26.3, unit=None, start=None, end=None, omop_table=None),        
]

####################################
# 3. Tokenize patient
batch: Dict[str, torch.Tensor] = tokenizer([ patient ], add_special_tokens=True, return_tensors='pt')
# > batch = {{
#     'input_ids': tensor([[ 5, 0, 7, 9, 27, 2049, 6557, 22433, 1]]), 
#     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
#     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
# }}
textual_tokens: List[str] = tokenizer.convert_events_to_tokens(patient)
# > textual_tokens = ['SNOMED/3950001', 'Gender/F', 'Ethnicity/Hispanic', 'SNOMED/609040007', 'LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0', 'SNOMED/12199005 || None || 26.0 - 28.899999618530273']

####################################
# 4. Run model
logits = model(**batch).logits
# > logits.shape = torch.Size([1, 9, 39818])

####################################
# 5. Get patient representation for finetuning (usually we choose the last token's logits)
representation = logits[:, -1, :]
# > representation.shape = torch.Size([1, 39818])
```

## Model Details

- **Developed by:** Shah lab @ Stanford University
- **Funded by:** Stanford Healthcare
- **Shared by:** Shah lab @ Stanford University
- **Model type:** {base_model}
- **Languages:** Electronic health record codes (as standardized by the [OMOP-CDM](https://ohdsi.github.io/CommonDataModel/index.html))
- **License:** CC-BY NC 4.0
- **Finetuned from model:** N/A -- trained from scratch

## Uses

This model is intended to generate representations for patients based on the structured data within their electronic health record. 
These representations can then be used for downstream tasks such as predicting diagnoses, detecting anomalies, or doing propensity score matching for causal inference.

### Direct Use

You will likely want to tune the model for your downstream use case. 

### Out-of-Scope Use

This model is for research purposes only. It is not for use in any real-world decision making that impacts patients, providers, or hospital operations.

## Bias, Risks, and Limitations

This model was trained on a corpus of 2 billion tokens sourced from 2.57 million patients from Stanford Medicine. 
The model will thus reflect the patterns of how care is delivered at Stanford Medicine, in addition to the racial and socioeconomic makeup of Stanford Medicine's patient base. 
This model may not generalize well to other hospitals and demographic mixes. 

While this is technically a generative model, we have not tested its generative abilities and thus do not anticipate it being used to generate synthetic EHR records. 
We aim to explore its generative abilities in future work.

## Training Details

Full training details are provided in our accompanying paper, [TODO]

### Training Data

The model is trained on 2 billion tokens sourced from 2.57 million patients from the [Stanford Medicine Research Data Repository (STARR)](https://academic.oup.com/jamiaopen/article/6/3/ooad054/7236015), 
which contains structured EHR data from both Stanford Health Care (primarily adult care) and Lucile Packard Children’s Hospital (primarily pediatric care). 
The dataset contains only structured data (i.e. no clinical text or images) and covers demographics (e.g. age, sex, race), diagnoses, procedures, laboratory results, medication prescriptions, and other coded clinical observations. 
The data is formatted according to the [Observational Medical Outcomes Partnership Common Data Model (OMOP-CDM)](https://ohdsi.github.io/CommonDataModel/cdm53.html). 
All data that we work with is deidentified.

### Training Procedure

We train our model using an autoregressive next code prediction objective, i.e. predict the next code in a patient's timeline given their previous codes.

## Citation

**BibTeX:**
```
@article{{TODO,
  title={{TODO}}, 
  author={{TODO}},
  booktitle={{TODO}},
  year={{TODO}}
}}
```

## Model Card Authors

Michael Wornow, Suhana Bedi, Ethan Steinberg
"""

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

    # Instantiate model and load weights
    new_state_dict = ckpt['state_dict']
    if 'gpt' in model_name:
        new_state_dict = {k.replace('model.', ''): v for k, v in new_state_dict.items()}
    elif 'hyena' in model_name:
        new_state_dict = {k.replace('model.', ''): v for k, v in new_state_dict.items()}
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
    path_to_local_dir = "./hf_model_for_upload"
    os.makedirs(path_to_local_dir, exist_ok=True)
    ## Model
    model.save_pretrained(path_to_local_dir, safe_serialization=False)
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
    api.upload_folder( folder_path=path_to_local_dir, repo_id=repo_id, )

    print("Model and tokenizer successfully pushed to the Hugging Face Hub!")