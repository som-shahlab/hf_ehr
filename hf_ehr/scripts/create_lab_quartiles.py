import json
from typing import List, Dict
import os
import numpy as np

from hf_ehr.config import PATH_TO_TOKENIZER_v9_DIR

def compute_quartiles(values):
    """Computes the quartiles of a list of numbers."""
    return np.percentile(values, [25, 50, 75])

def get_first_unit(units):
    """Simply retrieve the first unit in the list, even if it is None."""
    return units[0] if units else "No unit"

def process_data(data):
    output = {}
    print(f"Starting to process {len(data)} entries.")
    for code, details in data.items():
        values = details['values']
        ## TODO - fix unit identity function to something else
        units = details['units']
        unit = get_first_unit(units)
        quartiles = compute_quartiles(values)
        output[code] = {
            'quartiles': quartiles.tolist(),
            'unit': unit
        }
    return output

def main(input, output):
    with open(input, 'r') as file:
        data = json.load(file)
        print(f"Loaded data for {len(data.keys())} codes from the file.")
    
    result = process_data(data)
    
    with open(output, 'w') as file:
        json.dump(result, file, indent=4)
    
    print("DONE")

input_file = os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'lab_codes.json')  # Path to the input JSON file
output_file = os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'lab_code_2_quartiles.json')

if __name__ == '__main__':
    main(input_file, output_file)

