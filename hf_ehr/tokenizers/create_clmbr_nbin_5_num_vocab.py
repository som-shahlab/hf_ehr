import collections
import femr.datasets
import os
import json
import numpy as np
from tqdm import tqdm
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8
import time

# Load the FEMR dataset and PatientDatabase
start = time.time()
dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_v8, split='train', is_debug=False)
femr_db = femr.datasets.PatientDatabase(PATH_TO_FEMR_EXTRACT_v8)
print(f"Time to load FEMR database: {time.time() - start:.2f}s")

# Get the list of patient IDs
pids = dataset.get_pids().tolist()
print(f"Loaded n={len(pids)} patients from FEMRDataset using extract at: `{PATH_TO_FEMR_EXTRACT_v8}`")

# Collect lab values with progress bar
numericals = collections.defaultdict(list)

# Iterate over patient IDs and access the patient events using femr_db
for p_idx, pid in enumerate(tqdm(pids, desc="Processing patients")):
    # Access patient data by ID from femr_db
    patient = femr_db[pid]  # Retrieve patient by ID from femr_db
    if not hasattr(patient, 'events'):
        continue  # Skip if the patient has no events

    # Calculate weight based on the number of patients and events
    weight = 1 / (len(pids) * len(patient.events))
    
    # Process each event for the patient
    for event in patient.events:
        if (
            event.value is not None  # `value` is not None
            and (isinstance(event.value, float) or isinstance(event.value, int))  # `value` is numeric
        ):
            numericals[event.code].append((event.value, weight))

# Bin lab values with progress bar
n_bins: int = 5
output_data = []

for n_idx, (code, values) in enumerate(tqdm(numericals.items(), desc="Binning lab values")):
    quantiles = np.percentile([x[0] for x in values], np.linspace(0, 100, n_bins + 1))
    total_weight = np.sum([x[1] for x in values])
    weight_per_bucket = total_weight / n_bins

    for i in range(n_bins):
        # Prepare the output data in the specified format
        output_data.append({
            "code_string": code,
            "text_string": "",
            "type": 1,  # Assuming type 1 corresponds to "numerical_range"
            "val_start": quantiles[i],
            "val_end": quantiles[i + 1],
            "weight": weight_per_bucket
        })

# Save all the accumulated data to a single JSON file
final_output_path = "/share/pi/nigam/mwornow/hf_ehr/cache/create_cookbook/numerical_bins_final_output.json"
print(f"Saving all data to: {final_output_path}")
with open(final_output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print("Done processing and binning lab values.")
