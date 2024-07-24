import pandas as pd
import time
import json
from typing import Callable, List, Dict, Union
import os
from hf_ehr.config import PATH_TO_TOKENIZER_v8_DIR

def apply_in_parallel(path_to_code_2_numerical_info_parquet: str) -> Union[List, Dict]:
    path_to_intermediate_parquet: str = os.path.join(os.path.dirname(path_to_code_2_numerical_info_parquet), 'code_2_numerical_vocab.parquet')
    
    if os.path.exists(path_to_intermediate_parquet):
        df = pd.read_parquet(path_to_intermediate_parquet)
    else:
        start = time.time()
        print(f"Start | Loading PARQUET file from {path_to_code_2_numerical_info_parquet}...")
        code_2_numerical_info: pd.DataFrame = pd.read_parquet(path_to_code_2_numerical_info_parquet)
        print(f"Finish | Loading PARQUET file from {path_to_code_2_numerical_info_parquet}...")
        print(f"Time: {time.time() - start}s")

        print("Calculating Q1...")
        start = time.time()
        df_q1 = code_2_numerical_info.groupby(['codes', 'units'], dropna=False).agg({ 'values' : lambda x: x.quantile(0.25) }).reset_index().sort_values(['codes', 'units'])
        print(f"Time: {time.time() - start}s")
        print("Calculating Q2...")
        start = time.time()
        df_q2 = code_2_numerical_info.groupby(['codes', 'units'], dropna=False).agg({ 'values' : lambda x: x.quantile(0.5) }).reset_index().sort_values(['codes', 'units'])
        print(f"Time: {time.time() - start}s")
        print("Calculating Q3...")
        start = time.time()
        df_q3 = code_2_numerical_info.groupby(['codes', 'units'], dropna=False).agg({ 'values' : lambda x: x.quantile(0.75) }).reset_index().sort_values(['codes', 'units'])
        print("Calculating counts...")
        start = time.time()
        df_token_count = code_2_numerical_info.groupby(['codes', 'units'], dropna=False).size().reset_index(name='values').sort_values(['codes', 'units'])
        print(f"Time: {time.time() - start}s")
        df_q1['q1'] = df_q1['values']
        df_q1['q2'] = df_q2['values']
        df_q1['q3'] = df_q3['values']
        df_q1['token_count'] = df_token_count['values']
        df = df_q1.copy()
        df['units'] = df['units'].fillna('None')
        df['units'] = df['units'].astype(str)
        df.to_parquet(path_to_intermediate_parquet)
        assert df_q1.shape[0] == df_q2.shape[0] == df_q3.shape[0] == df_token_count.shape[0]
        assert (df_q1['codes'] != df_q2['codes']).sum() == 0
        assert (df_q1['codes'] != df_q3['codes']).sum() == 0
        assert (df_q1['codes'] != df_token_count['codes']).sum() == 0

    print("Creating `code_2_numerical_vocab`...")
    code_2_numerical_vocab = {}
    for idx, row in df.iterrows():
        code: str = row['codes']
        unit: str = row['units']
        q1: float = row['q1']
        q2: float = row['q2']
        q3: float = row['q3']
        
        # Setup
        if code not in code_2_numerical_vocab:
            code_2_numerical_vocab[code] = {}
        if 'unit_2_quartiles' not in code_2_numerical_vocab[code]:
            code_2_numerical_vocab[code]['unit_2_quartiles'] = {}

        # Save values
        code_2_numerical_vocab[code]['unit_2_quartiles'][unit] = [ q1, q2, q3, ]
    return code_2_numerical_vocab

if __name__ == '__main__':
    path_to_code_2_numerical_info_parquet = os.path.join(PATH_TO_TOKENIZER_v8_DIR, 'code_2_numerical_info.parquet')
    path_to_output: str = os.path.join(PATH_TO_TOKENIZER_v8_DIR, 'code_2_numerical_vocab.json')

    code_2_numerical_vocab = apply_in_parallel(path_to_code_2_numerical_info_parquet)
    
    # Save outputs
    with open(path_to_output, 'w') as fd:
        json.dump(code_2_numerical_vocab, fd, indent=2)