def readme_text(model_name, base_model, ctx_length, param_count):
    # "Run Model" section
    if 'llama' in model_name:
        section__run_model = f'''batch.pop("token_type_ids", None) # ! NOTE: Must remove 'token_type_ids' for llama model
logits = model(**batch).logits
# > logits.shape = torch.Size([1, 9, 39818])
'''
    elif 'hyena' in model_name:
        section__run_model = f'''batch.pop("token_type_ids", None) # ! NOTE: Must remove 'token_type_ids' for llama model
batch.pop("attention_mask", None) # ! NOTE: Must remove 'attention_mask' for hyena model
logits = model(**batch).logits
# > logits.shape = torch.Size([1, 9, 39818])
'''
    else:
        section__run_model = f'''logits = model(**batch).logits
# > logits.shape = torch.Size([1, 9, 39818])'''

    # README content
    return f"""
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

This is a **{base_model}** model with context length **{ctx_length}** with **{(param_count // 1e6):.2f}M** parameters from the [Context Clues paper](https://arxiv.org/abs/2412.16178).

It is a foundation model trained from scratch on the structured data within 2.57 million deidentified EHRs from Stanford Medicine.

As input, this model expects a sequence of coded medical events that have been mapped to Standard Concepts within the [OMOP-CDM vocabulary](https://ohdsi.github.io/CommonDataModel/index.html). As output, the model can generate either (a) synthetic future timelines or (b) a vector representation of a patient which can then be used for downstream prediction tasks.

## Usage

First, install the `hf-ehr` package:
```bash
pip install transformers torch hf-ehr
```

Second, run this Python script to do inference on a patient representation:

```python
from transformers import AutoModelForCausalLM
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from typing import List, Dict
import torch

####################################
# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("StanfordShahLab/{model_name}"{', trust_remote_code=True) # NOTE: Must trust remote code for Hyena' if 'hyena' in model_name else ')' }
tokenizer = CLMBRTokenizer.from_pretrained("StanfordShahLab/{model_name}")

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
{section__run_model}

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

Full training details are provided in our accompanying paper, [Context Clues](https://arxiv.org/abs/2412.16178).

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
@article{{wornow2024contextclues,
      title={{Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHRs}}, 
      author={{Michael Wornow and Suhana Bedi and Miguel Angel Fuentes Hernandez and Ethan Steinberg and Jason Alan Fries and Christopher Ré and Sanmi Koyejo and Nigam H. Shah}},
      year={{2024}},
      eprint={{2412.16178}},
      url={{https://arxiv.org/abs/2412.16178}}, 
}}
```

## Model Card Authors

Michael Wornow, Suhana Bedi, Ethan Steinberg
"""