from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Union
from hf_ehr.trainer.samplers import ApproxBatchSampler, SortishSampler
from omegaconf import DictConfig 
from hf_ehr.data.datasets import FEMRDataset, BaseDataset, AllTokensFEMRDataset
from hf_ehr.data.tokenization import BaseTokenizer, collate_femr_timelines
from loguru import logger
import numpy as np

def load_dataloaders(config: DictConfig, 
                     datasets: Dict[str, BaseDataset], 
                     tokenizer: BaseTokenizer) -> Dict[str, DataLoader]:
    dataset_name: str = config.data.dataset.name
    batch_size: Optional[int] = getattr(config.data.dataloader, 'batch_size', None)
    approx_batch_sampler: Optional[Any] = getattr(config.data.dataloader, 'approx_batch_sampler', None)
    dataloader_mode: str = getattr(config.data.dataloader, 'mode', 'batch')
    max_length: int = config.data.dataloader.max_length
    is_truncation_random: bool = config.data.dataloader.is_truncation_random
    is_mlm = False  # Assume non-MLM by default for GPT
    if 'model' in config and 'name' in config.model:
        if config.model.name == 'bert':
            is_mlm = True  # MLM is typically associated with BERT
    mlm_prob: float = config.data.mlm_prob if is_mlm else 0.0
    n_workers: int = config.data.dataloader.n_workers
    seed: int = config.main.seed
    n_replicas: int = len(config.trainer.devices)
    
    # Samplers
    if dataloader_mode == 'approx':
        logger.info("====> Loading ApproxBatchSampler")

        # Get sequence lengths for each example in dataset
        if dataset_name == 'FEMRDataset':
            # Each example in the dataset is a patient, so simply return the sequence length of each patient
            train_idx_to_seq_length: List[int] = tokenizer.get_seq_length_per_patient(datasets['train'])
            val_idx_to_seq_length: List[int] = tokenizer.get_seq_length_per_patient(datasets['val'])
            test_idx_to_seq_length: List[int] = tokenizer.get_seq_length_per_patient(datasets['test'])
        elif dataset_name == 'AllTokensFEMRDataset':
            # Each example in the dataset is a SUBSET of a patient, so return the sequence length of each example -- slightly trickier than FEMRDataset
            train_idx_to_seq_length: List[int] = datasets['train'].idx_to_seq_length
            val_idx_to_seq_length: List[int] = datasets['val'].idx_to_seq_length
            test_idx_to_seq_length: List[int] = datasets['test'].idx_to_seq_length
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
        # Train -- randomize (if desired) within/across batch sequence ordering
        train_bucket_size = approx_batch_sampler.bucket_size
        batch_max_tokens = approx_batch_sampler.max_tokens
        is_random_shuffle_across_buckets = approx_batch_sampler.is_random_shuffle_across_buckets
        is_random_shuffle_within_buckets = approx_batch_sampler.is_random_shuffle_within_buckets
        train_sort_sampler = SortishSampler(train_idx_to_seq_length, 
                                            train_bucket_size, 
                                            is_random_shuffle_across_buckets, 
                                            is_random_shuffle_within_buckets,
                                            n_replicas)
        train_batch_sampler = ApproxBatchSampler( train_idx_to_seq_length, train_sort_sampler, max_length, batch_max_tokens, )
        train_batch_sampler_kwargs = { 'batch_sampler' : train_batch_sampler, }
        # For val / test -- always sort by length, then execute in fixed sequence
        ## Val
        val_sort_sampler = SortishSampler( val_idx_to_seq_length, 1, False, False, n_replicas)
        val_batch_sampler = ApproxBatchSampler( val_idx_to_seq_length, val_sort_sampler, max_length, batch_max_tokens, )
        val_batch_sampler_kwargs = { 'batch_sampler' : val_batch_sampler, }
        ## Test
        test_sort_sampler = SortishSampler( test_idx_to_seq_length, 1, False, False, n_replicas)
        test_batch_sampler = ApproxBatchSampler( test_idx_to_seq_length, test_sort_sampler, max_length, batch_max_tokens, )
        test_batch_sampler_kwargs = { 'batch_sampler' : test_batch_sampler, }
    else:
        train_batch_sampler_kwargs = { 'batch_size' : batch_size, }
        val_batch_sampler_kwargs = { 'batch_size' : batch_size, }
        test_batch_sampler_kwargs = { 'batch_size' : batch_size, }

    train_loader = DataLoader(
        dataset=datasets['train'],
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, dataset_name, max_length, is_truncation_random, is_mlm, mlm_prob, seed),
        num_workers=n_workers,
        pin_memory=True,
        **train_batch_sampler_kwargs,
    )
    val_loader = DataLoader(
        dataset=datasets['val'],
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, dataset_name, max_length, is_truncation_random, is_mlm, mlm_prob, seed),
        num_workers=n_workers,
        pin_memory=True,
        **val_batch_sampler_kwargs,
    )
    test_loader = DataLoader(
        dataset=datasets['test'],
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, dataset_name, max_length, is_truncation_random, is_mlm, mlm_prob, seed),
        num_workers=n_workers,
        pin_memory=True,
        **test_batch_sampler_kwargs,
    )
    return {
        'train' : train_loader,
        'val' : val_loader,
        'test' : test_loader,
    }

def load_datasets(config: DictConfig, tokenizer: Optional[BaseTokenizer]) -> Dict[str, BaseDataset]:
    """Load all FEMR datasets. 
        - Takes ~8s for each dataset to load using /local-scratch/.
        - Takes ~8s for each dataset to load using /share/pi/.
    """
    dataset_name: str = config.data.dataset.name
    path_to_femr_extract: str = config.data.dataset.path_to_femr_extract
    is_debug: bool = getattr(config.data.dataset, 'is_debug', False)
    seed: int = config.main.seed
    
    # Load datasets
    if dataset_name == 'FEMRDataset':
        train_dataset = FEMRDataset(path_to_femr_extract, split='train', is_debug=is_debug, seed=seed)
        val_dataset = FEMRDataset(path_to_femr_extract, split='val', is_debug=is_debug, seed=seed)
        test_dataset = FEMRDataset(path_to_femr_extract, split='test', is_debug=is_debug, seed=seed)
    elif dataset_name == 'AllTokensFEMRDataset':
        max_length: int = config.data.dataloader.max_length
        assert tokenizer is not None, "Tokenizer must be provided for AllTokensFEMRDataset"
        train_dataset = AllTokensFEMRDataset(tokenizer, max_length, path_to_femr_extract, split='train', is_debug=is_debug, seed=seed)
        val_dataset = AllTokensFEMRDataset(tokenizer, max_length, path_to_femr_extract, split='val', is_debug=is_debug, seed=seed)
        test_dataset = AllTokensFEMRDataset(tokenizer, max_length, path_to_femr_extract, split='test', is_debug=is_debug, seed=seed)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    return { 
        'train' : train_dataset, 
        'val' : val_dataset, 
        'test' : test_dataset,
    }
