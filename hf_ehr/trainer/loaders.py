from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Union
from hf_ehr.trainer.samplers import ApproxBatchSampler, SortishSampler
from omegaconf import DictConfig 
from hf_ehr.data.datasets import FEMRDataset, collate_femr_timelines
from hf_ehr.data.tokenization import FEMRTokenizer, DescTokenizer
from loguru import logger
import numpy as np

def load_dataloaders(config: DictConfig, datasets: Dict[str, FEMRDataset], tokenizer: Union[FEMRTokenizer, DescTokenizer]) -> Dict[str, DataLoader]:
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
    path_to_code_2_detail: str =  config.data.tokenizer.path_to_code_2_detail
    
    sampling_strat: Optional[str] = config.data.sampling_strat
    sampling_kwargs: Optional[Dict] = config.data.sampling_kwargs
    excluded_vocabs = getattr(config.data.tokenizer, 'excluded_vocabs', None)
    is_remap_numerical_codes: bool = getattr(config.data.tokenizer, 'is_remap_numerical_codes', False)
    is_remap_codes_to_desc: bool = getattr(config.data.tokenizer, 'is_remap_codes_to_desc', False)
    is_clmbr: bool = getattr(config.data.tokenizer, 'is_clmbr', False)
    min_code_count: Optional[int] = config.data.tokenizer.min_code_count
    is_debug: bool = getattr(config.data.dataset, 'is_debug', False)
    seed: int = config.main.seed
    
    # Load datasets
    kwargs = {
        'sampling_strat' : sampling_strat,
        'sampling_kwargs' : sampling_kwargs,
        'excluded_vocabs' : excluded_vocabs,
        'is_remap_numerical_codes' : is_remap_numerical_codes,
        'is_remap_codes_to_desc' : is_remap_codes_to_desc,
        'is_clmbr' : is_clmbr,
        'min_code_count' : min_code_count,
        'is_debug' : is_debug,
        'seed' : seed,
    }
    train_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', **kwargs)
    val_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='val', **kwargs)
    test_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='test', **kwargs)
    
    return { 
            'train' : train_dataset, 
            'val' : val_dataset, 
            'test' : test_dataset,
    }
