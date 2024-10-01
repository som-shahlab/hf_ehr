import pandas as pd
from tqdm import tqdm
import tiktoken
from datasets import load_dataset
from collections import Counter
from itertools import islice

# Load the GPT-4 tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")

# Load the dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

# Function to count the occurrences of unique n-grams (general case)
def count_ngram_occurrences(text, n):
    # Tokenize the text using GPT-4 tiktoken tokenizer
    tokens = tokenizer.encode(text)
    
    # Generate n-grams from the tokenized data
    ngrams = zip(*(islice(tokens, i, None) for i in range(n)))
    
    # Count the occurrences of each n-gram using Counter
    ngram_counts = Counter(ngrams)
    
    # Return the counter dictionary of n-gram occurrences
    return ngram_counts


for i in [ 1, 2, 3, 4 ]:
    # Aggregate counts for all examples
    results = []
    for idx, example in tqdm(enumerate(ds['train']), total=len(ds['train'])):
        text = example['text']
        if text == '':
            # Skip blank examples
            continue
        counts = count_ngram_occurrences(text, i)
        for item, count in counts.items():
            results.append({
                'idx': idx,
                'n' : i,
                'count': count,
            })

    df = pd.DataFrame(results)
    df.to_parquet(f"wikitext-103--ngrams--{i}.parquet", index=False)