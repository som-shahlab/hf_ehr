"""
Usage:
    python hf_download.py

Purpose:
    Test if we can download an hf_ehr model+tokenizer from Hugging Face.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from typing import List, Dict
import torch
from tqdm import tqdm

models = [ 
    # 'gpt-base-512--clmbr',
    # 'gpt-base-1024--clmbr',
    # 'gpt-base-2048--clmbr',
    # 'gpt-base-4096--clmbr',
    # 'hyena-large-1024--clmbr',
    # 'hyena-large-4096--clmbr',
    # 'hyena-large-8192--clmbr',
    # 'hyena-large-16384--clmbr',
    # 'mamba-tiny-1024--clmbr',
    # 'mamba-tiny-4096--clmbr',
    # 'mamba-tiny-8192--clmbr',
    # 'mamba-tiny-16384--clmbr', 
    'llama-base-512--clmbr',
    'llama-base-1024--clmbr',
    'llama-base-2048--clmbr',
    'llama-base-4096--clmbr',
]

def run_test(model_name: str):
    model_name = model_name.replace('--', '-') # Remove the double dash

    ####################################
    # 1. Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(f"StanfordShahLab/{model_name}", trust_remote_code=True if 'hyena' in model_name else False)
    tokenizer = CLMBRTokenizer.from_pretrained(f"StanfordShahLab/{model_name}")

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
    # > batch = {
    #     'input_ids': tensor([[ 5, 0, 7, 9, 27, 2049, 6557, 22433, 1]]), 
    #     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
    #     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # }
    assert (batch['input_ids'] == torch.tensor([[ 5, 0, 7, 9, 27, 2049, 6557, 22433, 1]])).all(), f"batch['input_ids'] = {batch['input_ids']}"
    assert (batch['token_type_ids'] == torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])).all(), f"batch['token_type_ids'] = {batch['token_type_ids']}"
    assert (batch['attention_mask'] == torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])).all(), f"batch['attention_mask'] = {batch['attention_mask']}"

    textual_tokens: List[str] = tokenizer.convert_events_to_tokens(patient)
    # > textual_tokens = ['SNOMED/3950001', 'Gender/F', 'Ethnicity/Hispanic', 'SNOMED/609040007', 'LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0', 'SNOMED/12199005 || None || 26.0 - 28.899999618530273']
    assert len(textual_tokens) == 6, f"len(textual_tokens) = {len(textual_tokens)}"
    assert textual_tokens == ['SNOMED/3950001', 'Gender/F', 'Ethnicity/Hispanic', 'SNOMED/609040007', 'LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0', 'SNOMED/12199005 || None || 26.0 - 28.899999618530273'], f"textual_tokens = {textual_tokens}"
    
    ####################################
    # 4. Run model
    if 'llama' in model_name :
        batch.pop("token_type_ids", None)
    if 'hyena' in model_name:
        batch.pop("token_type_ids", None)
        batch.pop("attention_mask", None)
    logits = model(**batch).logits
    # > logits.shape = torch.Size([1, 9, 39818])
    assert logits.shape == torch.Size([1, 9, 39818]), f"logits.shape = {logits.shape}"

    ####################################
    # 5. Get patient representation for finetuning (usually we choose the last token's logits)
    repr = logits[:, -1, :]
    # > repr.shape = torch.Size([1, 39818])
    assert repr.shape == torch.Size([1, 39818]), f"repr.shape = {repr.shape}"

# Confirm each model works
good_models = []
bad_models = []
for model_name in tqdm(models):
    try:
        run_test(model_name)
        good_models.append(model_name)
    except Exception as e:
        bad_models.append(model_name)
        print(f"Error with model {model_name}: {e}")

print(f"Good models (n={len(good_models)}): {good_models}")
print(f"Bad models (n={len(bad_models)}): {bad_models}")
