from transformers import AutoModelForCausalLM
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from typing import List, Dict
import torch

####################################
# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("StanfordShahLab/gpt-base-512-clmbr")
tokenizer = CLMBRTokenizer.from_pretrained("StanfordShahLab/gpt-base-512-clmbr")

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
print(textual_tokens)
# > textual_tokens = ['SNOMED/3950001', 'Gender/F', 'Ethnicity/Hispanic', 'SNOMED/609040007', 'LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0', 'SNOMED/12199005 || None || 26.0 - 28.899999618530273']

####################################
# 4. Run model
outputs = model(**batch, output_hidden_states=True)

####################################
# 5. Get logits + probabilities for next token
logits = outputs.logits
print(logits.shape)
# > logits.shape = torch.Size([1, 9, 39818])
next_token_preds = torch.nn.functional.softmax(logits[:, -1, :], dim=-1) # should sum to 1
print(next_token_preds.shape)
# > next_token_pred.shape = torch.Size([1, 39818])

####################################
# 5. Get patient representation for finetuning (usually the hidden state of the LAST layer for the LAST token
last_layer_hidden_state = outputs.hidden_states[-1]
print(last_layer_hidden_state.shape)
# > last_layer_hidden_state.shape = torch.Size([1, 9, 768])
patient_rep = last_layer_hidden_state[:, -1, :]
print(patient_rep.shape)
# > patient_rep.shape = torch.Size([1, 768])