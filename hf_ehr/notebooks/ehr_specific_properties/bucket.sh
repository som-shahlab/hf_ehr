#!/bin/bash

# Define the values for task and strat
tasks=(
    "guo_los" 
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    # "chexpert" 
)

# Loop through each combination of task and strat
for task in "${tasks[@]}"; do
	python bucket.py --task "$task" --strat "$strat" --model clmbr
done

