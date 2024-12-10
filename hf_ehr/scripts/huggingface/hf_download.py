from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from typing import List, Dict
import torch

MODEL = 'gpt-base-512-clmbr'
MODEL = 'mamba-tiny-1024-clmbr'
MODEL = 'hyena-large-1024-clmbr'
MODEL = 'llama-base-512-clmbr'

####################################
# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(f"StanfordShahLab/{MODEL}")
tokenizer = CLMBRTokenizer.from_pretrained(f"StanfordShahLab/{MODEL}")

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
textual_tokens: List[str] = tokenizer.convert_events_to_tokens(patient)
# > textual_tokens = ['SNOMED/3950001', 'Gender/F', 'Ethnicity/Hispanic', 'SNOMED/609040007', 'LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0', 'SNOMED/12199005 || None || 26.0 - 28.899999618530273']

####################################
# 4. Run model
logits = model(**batch).logits
# > logits.shape = torch.Size([1, 9, 39818])

####################################
# 5. Get patient representation for finetuning (usually we choose the last token's logits)
repr = logits[:, -1, :]
# > repr.shape = torch.Size([1, 39818])
