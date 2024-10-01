import pandas as pd
import os

def compute_avg_brier_by_quartiles(df):
    # Group by model name, quartile, strat_metric, and optionally context_length, then calculate the average Brier score across all tasks
    group_columns = ['model_name', 'strat_metric', 'quartile']
    if 'context_length' in df.columns and df['context_length'].notnull().any():
        group_columns.append('context_length')

    avg_brier_scores = df.groupby(group_columns)['brier_score'].mean().reset_index()
    return avg_brier_scores

def main():
    # Define the directory containing the CSV files
    csv_dir = "/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/"
    
    # Explicit list of the files you want to include
    selected_files_1 = [
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',
    ]

    selected_files_2 = [
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__new_acutemi.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__new_hypertension.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last__new_pancan.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last__new_lupus.csv',
    ]

    selected_files_3 = [
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv',
        'metrics__gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',

    ]


    selected_files = [
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv'
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv'
        'metrics__hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_icu.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_los.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__guo_readmission.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_anemia.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyperkalemia.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hypoglycemia.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_hyponatremia.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__lab_thrombocytopenia.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_acutemi.csv',
        #'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_celiac.csv'
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hyperlipidemia.csv',
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_hypertension.csv',
        #'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_lupus.csv'
        'metrics__hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last__new_pancan.csv',

    ]

    # Initialize an empty DataFrame to hold all data
    all_data = pd.DataFrame()

    # Read each CSV file and append its data to the all_data DataFrame
    for filename in selected_files:
        csv_file = os.path.join(csv_dir, filename)
        df = pd.read_csv(csv_file)
        
        # Extract the correct model name from the filename
        base_name = os.path.basename(csv_file)
        model_name = base_name.split('__')[1] if len(base_name.split('__')) > 1 else base_name.split('__')[0]
        df['model_name'] = model_name
        
        # Check if context length exists in the filename and add it as a column if found
        context_length_parts = base_name.split('-')
        if len(context_length_parts) > 2 and context_length_parts[2].isdigit():
            context_length = int(context_length_parts[2])
            df['context_length'] = context_length
        else:
            df['context_length'] = None  # Handle files without context length
        
        # Extract task name from the filename
        task_name = base_name.split('__')[-1].replace(".csv", "")
        df['task'] = task_name
        
        # Append to the all_data DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)

    # Compute average Brier score by quartiles for each model, metric, and context length (if available), averaged over all tasks
    avg_brier_scores_by_quartiles = compute_avg_brier_by_quartiles(all_data)

    # Iterate over each unique model name, context length, and metric, and generate the tables
    for model_name in avg_brier_scores_by_quartiles['model_name'].unique():
        df_model = avg_brier_scores_by_quartiles[avg_brier_scores_by_quartiles['model_name'] == model_name]
        
        if 'context_length' in df_model.columns and df_model['context_length'].notnull().any():
            for context_length in df_model['context_length'].dropna().unique():
                df_context = df_model[df_model['context_length'] == context_length]
                for metric in df_context['strat_metric'].unique():
                    df_metric = df_context[df_context['strat_metric'] == metric]
                    table = df_metric[['quartile', 'brier_score']].copy()
                    table.columns = ['Quantile', 'Brier Score']

                    # Print the table for each model, context length, and metric
                    print(f"\nTable for Model '{model_name}', Metric '{metric}' at Context Length {context_length} (Averaged Over All Tasks):\n")
                    print(table.to_string(index=False))
        else:
            for metric in df_model['strat_metric'].unique():
                df_metric = df_model[df_model['strat_metric'] == metric]
                table = df_metric[['quartile', 'brier_score']].copy()
                table.columns = ['Quantile', 'Brier Score']

                # Print the table for each model and metric without context length
                print(f"\nTable for Model '{model_name}', Metric '{metric}' (Averaged Over All Tasks):\n")
                print(table.to_string(index=False))

if __name__ == "__main__":
    main()
