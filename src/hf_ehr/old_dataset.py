import glob
import femr.datasets
import transformers
import datasets
import json
import polars as pl
from typing import Callable
import collections
import multiprocessing
from typing import List, Tuple, Dict, Any
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel

PATH_TO_FEMR_EXTRACT: str = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8'

def multiprocess_func_on_femr_database(path_to_femr_extract: str, 
                                        path_to_output_dir: str,
                                        n_procs: int,
                                        map_func: Callable,
                                        map_args: Dict[str, Any],
                                        reduce_func: Callable,
                                        reduce_args: Dict[str, Any]) -> Tuple[Any, Any]:
    femr_db = femr.datasets.PatientDatabase(PATH_TO_FEMR_EXTRACT)
    pids: List[int] = sorted(list(femr_db.keys()))

    # Split patients across threads
    patient_ids_chunks: List[List[int]] = np.array_split(pids, n_procs)
    path_to_tmp_files: List[str] = [
        os.path.join(path_to_output_dir, f"proc_{i}") 
        for i in range(len(patient_ids_chunks))
    ]
    task_args: List[Tuple[str, List[int]]] = [
        (path_to_femr_extract, path_to_tmp_files[i], patient_ids_chunks[i], *map_args)
        for i in range(len(patient_ids_chunks))
    ]
    
    # Map
    with multiprocessing.get_context("forkserver").Pool(n_procs) as pool:
        map_output = list(tqdm(pool.imap(map_func, task_args), total=n_procs))

    # Reduce
    reduce_output = reduce_func(path_to_output_dir, path_to_tmp_files, *reduce_args)
    
    return map_output, reduce_output

def dump_timelines(path_to_femr_extract: str, path_to_output_file: str, patient_ids: List[str]):
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)

    # Collect data
    pl_data = collections.defaultdict(list)
    json_data = collections.defaultdict(list)
    for pid in patient_ids:
        events = femr_db[pid].events
        for e in events:
            pl_data['event_code'].append(e.code)
            pl_data['event_start'].append(e.start)
            pl_data['event_end'].append(e.end)
            pl_data['patient_id'].append(pid)
            json_data[pid].append(e.code)
    
    # Write to polars
    # Headers: patient_id |event_code | event_start | event_end
    df = pl.from_dict(pl_data)
    df.write_parquet(path_to_output_file + '.pq')
    
    # Write to JSONL
    # [key] = patient_id, [value] = list of codes (strings)
    with open(os.path.join(path_to_output_file + '.jsonl'), 'w') as fd:
        for pid, codes in json_data.items():
            fd.write(json.dumps({ 'pid' : pid, 'codes' : codes }) + '\n')
    
    return path_to_output_file

def join_timelines(path_to_output_dir: str, path_to_output_files: List[str]):
    # Join parquet
    df = pl.read_parquet(path_to_output_dir + '*.pq')
    df.write_parquet(os.path.join(path_to_output_dir, 'all_timelines.pq'))

    # Join JSONLs
    path_to_jsonl_files: List[str] = glob.glob(os.path.join(path_to_output_dir, '*.jsonl'))
    data = []
    for file in path_to_jsonl_files:
        with open(file, 'r') as fd:
            for line in fd:
                row = json.loads(line)
                data.append(row)
    with open(os.path.join(path_to_output_dir, 'all_timelines.jsonl'), 'w') as fd:
        for pid, codes in data.items():
            fd.write(json.dumps({ 'pid' : pid, 'codes' : codes }) + '\n')
    
    # Save as HF dataset
    dataset = datasets.Dataset.from_dict({
        'pid' : [ x['pid'] for x in data ],
        'timeline_text' : [ ' '.join(x['codes']) for x in data ],
    })
    dataset.save_to_disk(os.path.join(path_to_output_dir, 'all_timelines_hf.arrow'))

def batch_iterator(dataset, batch_size=1_000):                                                       
    for batch in dataset.iter(batch_size=batch_size):                                                
        yield batch["timeline_text"] 
    
    
if __name__ == '__main__':

    path_to_output_dir: str = '/share/pi/nigam/mwornow/tmp'
    n_procs: int = 40

    # Write all patient timelines to parquet file
    multiprocess_func_on_femr_database(PATH_TO_FEMR_EXTRACT, 
                                       path_to_output_dir,
                                       n_procs,
                                       map_func=dump_timelines,
                                       map_args={},
                                       reduce_func=join_timelines,
                                       reduce_args={})
    
    # Get list of all codes
    df = pl.read_parquet(os.path.join(path_to_output_dir, 'all_events.pq'))
    unique_codes: List[str] = sorted(df['code'].unique().tolist())
    code_2_int: Dict[str, int] = { idx: code for idx, code in enumerate(unique_codes) }
    json.dump(code_2_int, open(os.path.join(path_to_output_dir, 'code_2_int.json')))
    print("# of unique codes: ", len(unique_codes))

    # # Tokenize each JSONL file
    # path_to_jsonl_files: List[str] = glob.glob(os.path.join(path_to_output_dir, '*.jsonl'))
    # task_args: List[Tuple[str, List[int]]] = [
    #     (path_to_jsonl_files[i], code_2_int)
    #     for i in range(len(path_to_jsonl_files))
    # ]
    
    # # Map
    # with multiprocessing.get_context("forkserver").Pool(n_procs) as pool:
    #     output = list(tqdm(pool.imap(tokenize_jsonl, task_args), total=n_procs))
    
    # Train tokenizer on timelines
    dataset = datasets.load_dataset(os.path.join(path_to_output_dir, 'all_timelines_hf.arrow'))
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size: int = old_tokenizer.vocab_size
    print("Vocab size: ", vocab_size)

    # Train tokenizer
    new_tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(dataset), 
                                                          vocab_size, 
                                                          length=len(dataset))
    new_tokenizer.save_pretrained(os.path.join(path_to_output_dir, 'gpt2-tokenizer'))
    

    # Initialize empty GPT2 module
    config = GPT2Config()
    model = GPT2LMHeadModel(config)