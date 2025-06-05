import torch
from transformers import AutoModelForCausalLM
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from typing import List, Dict
import pandas as pd
import time
import argparse

# List of models to profile
models = [
    "StanfordShahLab/gpt-base-512-clmbr",
    "StanfordShahLab/gpt-base-1024-clmbr",
    "StanfordShahLab/gpt-base-2048-clmbr",
    "StanfordShahLab/gpt-base-4096-clmbr",
    "StanfordShahLab/llama-base-512-clmbr",
    "StanfordShahLab/llama-base-1024-clmbr",
    "StanfordShahLab/llama-base-2048-clmbr",
    "StanfordShahLab/llama-base-4096-clmbr",
    "StanfordShahLab/mamba-tiny-1024-clmbr",
    "StanfordShahLab/mamba-tiny-4096-clmbr",
    "StanfordShahLab/mamba-tiny-8192-clmbr",
    "StanfordShahLab/mamba-tiny-16384-clmbr",
    "StanfordShahLab/hyena-large-1024-clmbr",
    "StanfordShahLab/hyena-large-4096-clmbr",
    "StanfordShahLab/hyena-large-8192-clmbr",
    "StanfordShahLab/hyena-large-16384-clmbr",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=10, help="Number of forward passes to average over for timing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1. Define patient with max timeline
    full_patient: List[Event] = [
        Event(code='SNOMED/3950001', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='Gender/F', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='Ethnicity/Hispanic', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='SNOMED/609040007', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='LOINC/2236-8', value=-3.0,    unit=None, start=None, end=None, omop_table=None),
        Event(code='SNOMED/12199005', value=26.3,  unit=None, start=None, end=None, omop_table=None),   
        Event(code='SNOMED/3950001', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='Gender/F', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='Ethnicity/Hispanic', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='SNOMED/609040007', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='LOINC/2236-8', value=-3.0,    unit=None, start=None, end=None, omop_table=None),
        Event(code='SNOMED/12199005', value=26.3,  unit=None, start=None, end=None, omop_table=None),
        Event(code='SNOMED/609040007', value=None, unit=None, start=None, end=None, omop_table=None),
        Event(code='LOINC/2236-8', value=-3.0,    unit=None, start=None, end=None, omop_table=None),
        Event(code='SNOMED/12199005', value=26.3,  unit=None, start=None, end=None, omop_table=None),  
        Event(code='SNOMED/12199005', value=26.3,  unit=None, start=None, end=None, omop_table=None),      
        Event(code='SNOMED/12199005', value=26.3,  unit=None, start=None, end=None, omop_table=None),     
        Event(code='SNOMED/12199005', value=26.3,  unit=None, start=None, end=None, omop_table=None),      
    ] * 1000

    # 2. Profile each model
    stats = []
    fixed_length = 500 # Run every model at this seq length for uniform comparison
    for model_name in models:
        max_length: int = int(model_name.split("-")[-2]) - 20
        for seq_length in [fixed_length, max_length]:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(args.device)
                tokenizer = CLMBRTokenizer.from_pretrained(model_name)
                
                ####################################
                # 3. Truncate patient timeline
                patient = full_patient[:seq_length]

                ####################################
                # 4. Tokenize and move inputs to GPU
                batch: Dict[str, torch.Tensor] = tokenizer([patient], add_special_tokens=True, return_tensors='pt')
                batch = {k: v.to(args.device) for k, v in batch.items()}
                if 'hyena' in model_name:
                    batch.pop("token_type_ids", None)
                    batch.pop("attention_mask", None)


                ####################################
                # 5. Reset and profile peak memory
                torch.cuda.empty_cache()                       # clear any cached memory
                torch.cuda.reset_peak_memory_stats(args.device)     # reset the peak‐memory counter

                ####################################
                # 5. Run multiple forward passes
                start = time.time()
                for _ in range(args.n_trials):
                    logits = model(**batch).logits  # shape: [1, seq_len, vocab_size]
                end = time.time()
                avg_time = (end - start) / args.n_trials

                ####################################
                # 6. Save peak GPU memory (in MB)
                peak_bytes = torch.cuda.max_memory_allocated(args.device)
                peak_megabytes = peak_bytes / (1024 ** 2)
                
                ####################################
                # 7. Save stats
                print(f"Model: {model_name} | seq_len: {seq_length} | Peak memory: {peak_megabytes:.2f} MB | Time/patient: {avg_time:.2f} s")
                stats.append({
                    "model_name": model_name,
                    "peak_memory": peak_megabytes,
                    "seq_length": seq_length,
                    "time_per_seq": avg_time,
                })
                
                del model, tokenizer, batch, logits
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error profiling {model_name}: {e}")
                stats.append({
                    "model_name": model_name,
                    "peak_memory": None,
                    "seq_length": seq_length,
                    "time_per_seq": None,
                })

    df = pd.DataFrame(stats)
    df.to_csv("inference_stats.csv", index=False)
    
    print("=== Fixed length ===")
    print(df[df['seq_length'] == fixed_length].to_markdown(index=False))
    print("=== Max length ===")
    print(df[df['seq_length'] != fixed_length].to_markdown(index=False))
    