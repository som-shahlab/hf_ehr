
"""
import json
import time
from typing import List, Dict
import os
from tqdm import tqdm
from hf_ehr.config import (
    TokenizerConfigEntry, CodeTCE, 
    CountOccurrencesTCEStat, PPLTCEStat,
    save_tokenizer_config_to_path,
    PATH_TO_TOKENIZER_CLMBR_v8_DIR
)

PATH_TO_CLMBR_JSON: str = os.path.join(PATH_TO_TOKENIZER_CLMBR_v8_DIR, 'clmbr_v8_original_dictionary.json')

if __name__ == '__main__':
    start_total = time.time()
    
    # Load original CLMBR dictionary
    clmbr: Dict[str, List] = json.load(open(PATH_TO_CLMBR_JSON))
    path_to_output_dir: str = os.path.dirname('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizers/clmbr_v8_nbin_0/')

    tokenizer_config: List[TokenizerConfigEntry] = []
    for token in tqdm(clmbr['regular'], desc='Looping thru CLMBR codes...', total=len(clmbr['regular'])):
        code: str = token['code_string']
        type_ = token['type']
        
        # Skip ignored tokens
        if type_ == 'unused':
            continue

        # All tokens (numerical or categorical) are stored as 'code'
        defaults = {
            'code' : code,
            'description' : None,
            'type' : 'code',
            'stats' : [
                # dummy values, just to show what's possible
                CountOccurrencesTCEStat(split='train', dataset='v8'),
                PPLTCEStat(split='train', dataset='v8', model="gpt2-base-1024"),
            ],
        }

        # Create new token as a CodeTCE
        new_token = CodeTCE(
            **defaults,
        )
        
        tokenizer_config.append(new_token)
        
    path_to_output_file: str = os.path.join(path_to_output_dir, 'tokenizer_config.json')
    print(f"Saving CLMBR vocab to: `{path_to_output_file}`")
    save_tokenizer_config_to_path(path_to_output_file, tokenizer_config)
    
    n_new_tokens: int = len(tokenizer_config)
    n_old_tokens: int = len([ x for x in clmbr['regular'] if x['type'] != 'unused' ])
    print("Number of tokens in new CLMBR vocab: ", n_new_tokens)
    print("Number of tokens in old CLMBR vocab: ", n_old_tokens)
    assert n_new_tokens == n_old_tokens, f"ERROR - Mismatch in vocab lengths"
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print("Done!")
"""
import collections
import femr.datasets
import os
import json
import numpy as np
from tqdm import tqdm
import time
from typing import List, Dict
from hf_ehr.config import (
    TokenizerConfigEntry, NumericalRangeTCE, CategoricalTCE, CodeTCE, 
    CountOccurrencesTCEStat, PPLTCEStat,
    save_tokenizer_config_to_path,
    PATH_TO_TOKENIZER_CLMBR_v8_DIR
)
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8

# Path to original CLMBR dictionary
PATH_TO_CLMBR_JSON: str = os.path.join(PATH_TO_TOKENIZER_CLMBR_v8_DIR, 'clmbr_v8_original_dictionary.json')

# Load the FEMR dataset and PatientDatabase
start = time.time()
dataset = femr.datasets.FEMRDataset(PATH_TO_FEMR_EXTRACT_v8, split='train', is_debug=False)
femr_db = femr.datasets.PatientDatabase(PATH_TO_FEMR_EXTRACT_v8)
print(f"Time to load FEMR database: {time.time() - start:.2f}s")

# Load original CLMBR dictionary
clmbr: Dict[str, List] = json.load(open(PATH_TO_CLMBR_JSON))
path_to_output_dir: str = os.path.dirname(PATH_TO_CLMBR_JSON)

# Get the list of patient IDs
pids = dataset.get_pids().tolist()
print(f"Loaded n={len(pids)} patients from FEMRDataset using extract at: `{PATH_TO_FEMR_EXTRACT_v8}`")

# Collect lab values with progress bar (binning)
n_bins: int = 10
numericals = collections.defaultdict(list)
for p_idx, pid in enumerate(tqdm(pids, desc="Processing patients")):
    patient = femr_db[pid]  # Retrieve patient by ID from femr_db
    if not hasattr(patient, 'events'):
        continue  # Skip if the patient has no events
    
    for event in patient.events:
        if event.value is not None and (isinstance(event.value, float) or isinstance(event.value, int)):  # Numeric
            numericals[event.code].append(event.value)

# Bin the lab values (using quantiles)
binned_numericals = {}
for code, values in tqdm(numericals.items(), desc="Binning lab values"):
    quantiles = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    binned_numericals[code] = quantiles.tolist()  # Convert ndarray to list

# Create the tokenizer config
tokenizer_config: List[TokenizerConfigEntry] = []
for token in tqdm(clmbr['regular'], desc='Looping thru CLMBR codes...', total=len(clmbr['regular'])):
    code: str = token['code_string']
    val_start: float = token['val_start']
    val_end: float = token['val_end']
    type_ = token['type']
    text_string = token['text_string']
    
    # Skip ignored tokens
    if type_ == 'unused':
        continue

    defaults = {
        'code': code,
        'description': None,
        'type': (
            'numerical_range' if type_ == 'numeric' else
            'categorical' if type_ == 'text' else
            'code'
        ),
        'stats': [
            CountOccurrencesTCEStat(split='train', dataset='v8'),
            PPLTCEStat(split='train', dataset='v8', model="gpt2-base-1024"),
        ],
    }
    
    if type_ == 'code':
        new_token = CodeTCE(**defaults)
    elif type_ == 'text':
        new_token = CategoricalTCE(tokenization={'categories': [text_string]}, **defaults)
    elif type_ == 'numeric':
        unit: str = "None"
        # Use the binned quantiles for numeric tokens
        if code in binned_numericals:
            quantiles = binned_numericals[code]
            new_token = NumericalRangeTCE(
                tokenization={
                    'unit': unit,
                    'range_start': quantiles[0],  # Start of the range
                    'range_end': quantiles[-1],   # End of the range
                    'bins': quantiles  # The actual quantiles
                },
                **defaults,
            )
        else:
            # Fallback to original start/end values if no binned data
            new_token = NumericalRangeTCE(
                tokenization={
                    'unit': unit,
                    'range_start': val_start,
                    'range_end': val_end,
                },
                **defaults,
            )
    else:
        raise ValueError(f"ERROR - Unknown type for code {code}: {type_}")

    tokenizer_config.append(new_token)

# Save the tokenizer config
path_to_output_file: str = os.path.join(path_to_output_dir, 'tokenizer_config.json')
print(f"Saving CLMBR vocab to: `{path_to_output_file}`")
save_tokenizer_config_to_path(path_to_output_file, tokenizer_config)

n_new_tokens: int = len(tokenizer_config)
n_old_tokens: int = len([x for x in clmbr['regular'] if x['type'] != 'unused'])
print("Number of tokens in new CLMBR vocab: ", n_new_tokens)
print("Number of tokens in old CLMBR vocab: ", n_old_tokens)
assert n_new_tokens == n_old_tokens, f"ERROR - Mismatch in vocab lengths"

print(f"Total time taken: {round(time.time() - start, 2)}s")
print("Done!")
