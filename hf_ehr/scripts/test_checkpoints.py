import os
import re
from tqdm import tqdm
import torch

def rename_files(file_paths):
    """Remove -true_val=XXXX from the file name and rename the file if it already exists"""
    # Regex pattern to match the "-true_val=XXXX" phrase where XXXX is an integer
    pattern = re.compile(r'-true_val=\d+')
    
    for file_path in tqdm(file_paths):
        if not os.path.exists(file_path): 
            continue

        # Substitute the pattern with an empty string
        new_file_path = re.sub(pattern, '', file_path)
        
        # Skip if the file is already renamed
        if file_path == new_file_path:
            continue
        
        # Rename the file if the new name is different from the old name
        if os.path.exists(new_file_path):
            # From 'train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt' to 1000000000
            ckpt_val: int = int(re.search(r'ckpt_val=(\d+)', file_path).group(1))
            # Keep whichever checkpoint is closer to ckpt_val
            ckpt_old = torch.load(new_file_path, map_location='cpu')
            ckpt_new = torch.load(file_path, map_location='cpu')
            if abs(ckpt_old['train_total_tokens_nonPAD'] - ckpt_val) < abs(ckpt_new['train_total_tokens_nonPAD'] - ckpt_val):
                # Old checkpoint is closer to ckpt_val, so keep it
                print(f"WARNING: File already exists, skipping rename @ {file_path}")
                print(f"ckpt_val: {ckpt_val}, ckpt_old: {ckpt_old['train_total_tokens_nonPAD']}, ckpt_new: {ckpt_new['train_total_tokens_nonPAD']}")
                # Remove bad checkpoint
                os.remove(file_path)
                continue
            else:
                print(f"WARNING: File already exists, but new ckpt is closer to ckpt_val, so renaming @ {file_path}")
                print(f"ckpt_val: {ckpt_val}, ckpt_old: {ckpt_old['train_total_tokens_nonPAD']}, ckpt_new: {ckpt_new['train_total_tokens_nonPAD']}")
                # Remove bad checkpoint
                os.remove(new_file_path)
        os.rename(file_path, new_file_path)


def list_matching_files(dir, ckpt_val: int = '*'):
    output = []
    for model in ['bert-base', 'gpt-base', 'hyena-large', 'mamba-tiny', 'llama-base']:
        print("Searching for", model)
        output += os.popen(f'ls /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/{model}-*--clmbr/ckpts/train-tokens-total_nonPAD-true_val=*-ckpt_val={ckpt_val}-persist.ckpt').read().splitlines()
    # CLMBR-k
    output += os.popen(f'ls /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-*--clmbr_*k/ckpts/train-tokens-total_nonPAD-true_val=*-ckpt_val={ckpt_val}-persist.ckpt').read().splitlines()
    return output

def get_max_token_ckpts():
    output = []
    for model in ['bert-base', 'gpt-base', 'hyena-large', 'mamba-tiny', 'llama-base']:
        print("Searching for", model)
        output += os.popen(f'ls /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/{model}-*--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=*-persist.ckpt').read().splitlines()
    # CLMBR-k
    output += os.popen(f'ls /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-*--clmbr_*k/ckpts/train-tokens-total_nonPAD-ckpt_val=*-persist.ckpt').read().splitlines()
    # Loop through all files and keep only the one with the highest ckpt_val per model
    max_ckpts = {}
    max_ckpt_paths = {}
    for file_path in output:
        # From 'train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt' to 1000000000
        ckpt_val: int = int(re.search(r'ckpt_val=(\d+)', file_path).group(1))
        # Part [MODEL] from '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/[MODEL]/ckpts/train-tokens-total_nonPAD-ckpt_val=*-persist.ckpt'
        model = re.search(r'runs_backup/([a-z-0-9\-_A-Z]+)', file_path).group(1)
        if model not in max_ckpts or max_ckpts[model] < ckpt_val:
            max_ckpts[model] = ckpt_val
            max_ckpt_paths[model] = file_path
    return max_ckpts, max_ckpt_paths

max_ckpts, max_ckpt_paths = get_max_token_ckpts()
for key, val in max_ckpts.items():
    # Print key to 1 decimal place
    print(f"{key}: {int(val)/1e9:.1f}B | Path: {max_ckpt_paths[key]}")
exit()

paths = list_matching_files('/share/pi/nigam/suhana/hf_ehr/cache/runs_backup')
print(f"Found {len(paths)} files")
print(f"Paths: {paths}")
rename_files(paths)
