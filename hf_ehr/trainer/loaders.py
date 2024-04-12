from torch.utils.data import DataLoader
from typing import Any, Dict, Optional
from hf_ehr.trainer.samplers import ApproxBatchSampler, SortishSampler
from omegaconf import DictConfig 
from hf_ehr.data.datasets import FEMRDataset, FEMRTokenizer, collate_femr_timelines
from loguru import logger
from collections import OrderedDict
import numpy as np
from random import shuffle
import torch

def load_dataloaders(config: DictConfig, datasets: Dict[str, FEMRDataset], tokenizer: FEMRTokenizer) -> Dict[str, DataLoader]:
    batch_size: Optional[int] = getattr(config.data.dataloader, 'batch_size', None)
    approx_batch_sampler: Optional[Any] = getattr(config.data.dataloader, 'approx_batch_sampler', None)
    max_length: int = config.data.dataloader.max_length
    is_truncation_random: bool = config.data.dataloader.is_truncation_random
    is_mlm: bool = config.model.name == 'bert'
    mlm_prob: float = config.data.mlm_prob if is_mlm else 0.0
    n_workers: int = config.data.dataloader.n_workers
    seed: int = config.main.seed
    n_replicas: int = len(config.trainer.devices)
    
    # Samplers
    if approx_batch_sampler is not None:
        logger.info("====> Loading ApproxBatchSampler")
        # Train -- randomize (if desired) within/across batch sequence ordering
        train_bucket_size = approx_batch_sampler.bucket_size
        batch_max_tokens = approx_batch_sampler.max_tokens
        is_random_shuffle_across_buckets = approx_batch_sampler.is_random_shuffle_across_buckets
        is_random_shuffle_within_buckets = approx_batch_sampler.is_random_shuffle_within_buckets
        train_sort_sampler = SortishSampler(datasets['train'].get_seq_lengths(), 
                                            train_bucket_size, 
                                            is_random_shuffle_across_buckets, 
                                            is_random_shuffle_within_buckets,
                                            n_replicas)
        train_batch_sampler = ApproxBatchSampler( datasets['train'].get_seq_lengths(), train_sort_sampler, max_length, batch_max_tokens, )
        train_batch_sampler_kwargs = { 'batch_sampler' : train_batch_sampler, }
        # For val / test -- always sort by length, then execute in fixed sequence
        ## Val
        val_sort_sampler = SortishSampler( datasets['val'].get_seq_lengths(), 1, False, False, n_replicas)
        val_batch_sampler = ApproxBatchSampler( datasets['val'].get_seq_lengths(), val_sort_sampler, max_length, batch_max_tokens, )
        val_batch_sampler_kwargs = { 'batch_sampler' : val_batch_sampler, }
        ## Test
        test_sort_sampler = SortishSampler( datasets['test'].get_seq_lengths(), 1, False, False, n_replicas)
        test_batch_sampler = ApproxBatchSampler( datasets['test'].get_seq_lengths(), test_sort_sampler, max_length, batch_max_tokens, )
        test_batch_sampler_kwargs = { 'batch_sampler' : test_batch_sampler, }
    else:
        train_batch_sampler_kwargs = { 'batch_size' : batch_size, }
        val_batch_sampler_kwargs = { 'batch_size' : batch_size, }
        test_batch_sampler_kwargs = { 'batch_size' : batch_size, }

    train_loader = DataLoader(
        dataset=datasets['train'],
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, is_mlm, mlm_prob, seed),
        num_workers=n_workers,
        pin_memory=True,
        **train_batch_sampler_kwargs,
    )
    val_loader = DataLoader(
        dataset=datasets['val'],
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, is_mlm, mlm_prob, seed),
        num_workers=n_workers,
        pin_memory=True,
        **val_batch_sampler_kwargs,
    )
    test_loader = DataLoader(
        dataset=datasets['test'],
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, is_mlm, mlm_prob, seed),
        num_workers=n_workers,
        pin_memory=True,
        **test_batch_sampler_kwargs,
    )
    return {
        'train' : train_loader,
        'val' : val_loader,
        'test' : test_loader,
    }

def load_datasets(config: DictConfig) -> Dict[str, FEMRDataset]:
    """Load all FEMR datasets. 
        - Takes ~8s for each dataset to load using /local-scratch/.
        - Takes ~8s for each dataset to load using /share/pi/.
    """
    path_to_femr_extract: str = config.data.dataset.path_to_femr_extract
    sampling_strat: Optional[str] = config.data.sampling_strat
    sampling_kwargs: Optional[Dict] = config.data.sampling_kwargs
    is_debug: bool = getattr(config.data.dataset, 'is_debug', False)
    train_dataset = FEMRDataset(path_to_femr_extract, split='train', sampling_strat=sampling_strat, sampling_kwargs=sampling_kwargs, is_debug=is_debug)
    val_dataset = FEMRDataset(path_to_femr_extract, split='val', sampling_strat=sampling_strat, sampling_kwargs=sampling_kwargs, is_debug=is_debug)
    test_dataset = FEMRDataset(path_to_femr_extract, split='test', sampling_strat=sampling_strat, sampling_kwargs=sampling_kwargs, is_debug=is_debug)
    
    return { 
            'train' : train_dataset, 
            'val' : val_dataset, 
            'test' : test_dataset,
    }