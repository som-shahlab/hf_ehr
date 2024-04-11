from torch.utils.data import DataLoader
from typing import Dict, Optional
from hf_ehr.hf_ehr.trainer.samplers import ApproxBatchSampler, SortishSampler
from omegaconf import DictConfig 
from hf_ehr.data.datasets import FEMRDataset, FEMRTokenizer, collate_femr_timelines

from collections import OrderedDict
import numpy as np
from random import shuffle
import torch

def load_dataloaders(config: DictConfig, datasets: Dict[str, FEMRDataset], tokenizer: FEMRTokenizer) -> Dict[str, DataLoader]:
    batch_size: Optional[int] = config.data.dataloader.batch_size
    batch_max_tokens: Optional[int] = config.data.dataloader.batch_max_tokens
    max_length: int = config.data.dataloader.max_length
    is_truncation_random: bool = config.data.dataloader.is_truncation_random
    n_workers: int = config.data.dataloader.n_workers
    seed: int = config.main.seed
    n_replicas: int = torch.cuda.device_count()
    rank: int = torch.distributed.get_rank()
    
    # Samplers
    if batch_max_tokens:
        train_sort_sampler = SortishSampler( datasets['train'].get_seq_lengths(), 1_000, n_replicas, rank, )
        train_batch_sampler = ApproxBatchSampler( datasets['train'].get_seq_lengths(), train_sort_sampler, batch_max_tokens, )
        val_sort_sampler = SortishSampler( datasets['val'].get_seq_lengths(), 1, n_replicas, rank, )
        val_batch_sampler = ApproxBatchSampler( datasets['val'].get_seq_lengths(), val_sort_sampler, batch_max_tokens, )
        test_sort_sampler = SortishSampler( datasets['test'].get_seq_lengths(), 1, n_replicas, rank, )
        test_batch_sampler = ApproxBatchSampler( datasets['test'].get_seq_lengths(), test_sort_sampler, batch_max_tokens, )
    else:
        train_batch_sampler, val_batch_sampler, test_batch_sampler = None, None, None

    train_loader = DataLoader(
        dataset=datasets['train'],
        batch_size=batch_size,
        batch_sampler=train_batch_sampler,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=datasets['val'],
        batch_size=batch_size,
        batch_sampler=val_batch_sampler,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=datasets['test'],
        batch_size=batch_size,
        batch_sampler=test_batch_sampler,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
        pin_memory=True,
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

    train_dataset = FEMRDataset(path_to_femr_extract, split='train')
    val_dataset = FEMRDataset(path_to_femr_extract, split='val')
    test_dataset = FEMRDataset(path_to_femr_extract, split='test')
    
    return { 
            'train' : train_dataset, 
            'val' : val_dataset, 
            'test' : test_dataset,
    }