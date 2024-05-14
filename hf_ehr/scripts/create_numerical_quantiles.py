import json
from typing import List, Dict, Union
import os
import numpy as np
from hf_ehr.config import PATH_TO_TOKENIZER_v9_DIR
from tqdm import tqdm
import collections

def compute_quartiles(values: Union[np.ndarray, List[float]]) -> List[float]:
    """Computes the quartiles of a list of numbers."""
    return list(np.percentile(values, [25, 50, 75]))

def get_first_unit(units):
    """Simply retrieve the first unit in the list, even if it is None."""
    return units[0] if units else "No unit"

def create_numerical_vocab_codes(path_to_code_2_numerical_info_json: str):
    """Given the path to the `code_2_numerical_info.json` file, this calculates quantiles
        for each numerical values, then converts to vocab tokens."""
    code_2_numerical_info: Dict[str, Dict] = json.load(open(path_to_code_2_numerical_info_json, 'r'))
    for code, info in tqdm(code_2_numerical_info.items(), desc='Looping through code_2_numerical_info...'):
        units: List[str] = info['units']
        values: List[float] = info['values']

        # Group values by unit
        unit_2_values: Dict[str, List[float]] = collections.defaultdict(list)
        for (u, v) in zip(units, values):
            unit_2_values[u].append(v)

        # For each unit's values, calculate quartiles within that unit
        unit_2_quartiles: Dict[str, List[float]] = collections.defaultdict(list)
        for (u, values) in unit_2_values.items():
            quartiles = compute_quartiles(values)
            unit_2_quartiles[u] = quartiles
        
        # Count how many occurrences each new token will have
        token_2_count = collections.defaultdict(int)
        for (u, quartiles) in unit_2_quartiles.items():
            for q_idx in range(len(quartiles) + 1):
                token: str = f"{code} | {u} | Q{q_idx + 1}" # "STANFORD_OBS/123 | mmol | Q4"
                if q_idx == 0:
                    # Lower-most part of range
                    token_2_count[token] = sum([ v <= quartiles[0] for v in unit_2_values[u] ])
                elif q_idx == len(quartiles):
                    # Higher-most part of range
                    token_2_count[token] = sum([ quartiles[-1] < v for v in unit_2_values[u] ])
                else:
                    # Middle part of range
                    token_2_count[token] = sum([ quartiles[q_idx - 1] < v <= quartiles[q_idx] for v in unit_2_values[u] ])
        assert len(token_2_count) == len(unit_2_quartiles) * 4

        # Save new vocab
        code_2_numerical_info[code]['unit_2_quartiles'] = dict(unit_2_quartiles)
    return code_2_numerical_info

if __name__ == '__main__':
    path_to_code_2_numerical_info_json = os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'code_2_numerical_info.json')
    code_2_numerical_info = create_numerical_vocab_codes(path_to_code_2_numerical_info_json)

    with open(path_to_code_2_numerical_info_json, 'w') as fd:
        json.dump(code_2_numerical_info, fd, indent=2)
