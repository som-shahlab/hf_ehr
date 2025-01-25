"""

Usage:
    python3 val_ppl_nlp.py --model_name gpt2 --dataset_name Salesforce/wikitext --dataset_version wikitext-103-raw-v1
    
    python3 val_ppl_nlp.py --model_name gpt2 --dataset_name hazyresearch/LoCoV1-Documents --dataset_version '' --dataset_split 'test' --text_column_name 'passage'
"""

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import numpy as np
from collections import defaultdict
import argparse
from typing import List

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--dataset_name', type=str, default='Salesforce/wikitext')
    parser.add_argument('--dataset_version', type=str, default='wikitext-103-raw-v1')
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--text_column_name', type=str, default='text')
    parser.add_argument('--max_pos', type=int, default=10_000)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--y_lim', type=int, default=200)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load GPT-2 model and tokenizer
    model_name = args.model_name
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = 'cuda'
    model.to(device)
    model.eval()

    # Load a dataset from Hugging Face
    dataset = load_dataset(args.dataset_name, args.dataset_version, split=args.dataset_split)

    # Dictionary to store perplexities by position
    position_perplexities = defaultdict(list)

    # Load most recent position_perplexities.pkl
    is_resuming, resume_idx = False, 0
    files = sorted([ x for x in os.listdir('./') if x.startswith('position_perplexities_') and x.endswith('.pkl') ], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(files) > 0:
        print(f'Resuming from file `{files[-1]}`')
        position_perplexities = pickle.load(open(files[-1], 'rb'))
        is_resuming = True
        resume_idx = int(files[-1].split('_')[-1].split('.')[0])

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        # Skip if resuming and not at the correct index
        if is_resuming and idx < resume_idx:
            continue
        # Save every 10,000 examples
        if idx % 100 == 0:
            pickle.dump(position_perplexities, open(f'position_perplexities_{idx}.pkl', 'wb'))

        # Skip blank examples
        if example[args.text_column_name] == '':
            continue

        # Tokenize the full text
        tokens = tokenizer(example[args.text_column_name], return_tensors="pt")
        input_ids = tokens.input_ids.to(device)

        # Process the sequence in sliding windows
        stride: int = args.stride
        max_length: int = args.max_length
        seq_len: int = input_ids.size(1)
        
        all_token_perplexities: List[int] = []
        for start_pos in range(0, seq_len, stride):
            end_pos = min(start_pos + max_length, seq_len)
            overflow = start_pos + max_length - end_pos # number of tokens to ignore
            chunk_ids = input_ids[:, start_pos:end_pos]
            
            with torch.no_grad():
                outputs = model(chunk_ids, labels=chunk_ids)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = chunk_ids[..., 1:].contiguous()
                token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                      shift_labels.view(-1))
            
            # Convert token losses to perplexity
            chunk_perplexities = torch.exp(token_losses).cpu().tolist()
            
            # For the first window, keep all tokens
            if start_pos == 0:
                all_token_perplexities.extend(chunk_perplexities)
            # For subsequent windows, only keep tokens from end position -stride_length onward
            else:
                all_token_perplexities.extend(chunk_perplexities[-(stride - overflow):])
            
            if end_pos >= seq_len:
                break
        assert len(all_token_perplexities) == seq_len - 1, f"len(all_token_perplexities) == {len(all_token_perplexities)} != seq_len - 1 == {seq_len - 1}"

        # Store perplexities by position
        for pos, ppl in enumerate(all_token_perplexities):
            position_perplexities[pos].append(ppl)
    pickle.dump(position_perplexities, open(f'position_perplexities_{idx}.pkl', 'wb'))

    # Filter out positions with less than 30 samples
    position_perplexities = {pos: values for pos, values in position_perplexities.items() if len(values) >= 30}

    # Calculate median perplexity for each position
    positions = sorted(position_perplexities.keys())[:args.max_pos]
    median_perplexities = [np.median(position_perplexities[pos]) for pos in positions]

    df = pd.DataFrame([ { 'position': pos, 'median': ppl } for pos, ppl in enumerate(median_perplexities) ])
    df['ema_median'] = df['median'].ewm(span=20, adjust=False).mean()
    df.to_csv(f'{args.model_name}-{args.dataset_name}-{args.dataset_version}-{args.dataset_split}_perplexity_plot.csv'.replace('/', '_'), index=False)
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(positions, df['ema_median'])
    plt.xlabel('Token Position')
    plt.ylabel('Median PPL')
    plt.title('Token position v. Median PPL (EMA)')
    plt.grid(True)
    plt.ylim(0, args.y_lim)
    plt.savefig(f'{args.model_name}-{args.dataset_name}-{args.dataset_version}-{args.dataset_split}_perplexity_plot.png'.replace('/', '_'))