from torch.utils.data import DataLoader
from typing import Dict
from omegaconf import DictConfig 
from hf_ehr.data.datasets import FEMRDataset, FEMRTokenizer, collate_femr_timelines

def load_dataloaders(config: DictConfig, datasets: Dict[str, FEMRDataset], tokenizer: FEMRTokenizer) -> Dict[str, DataLoader]:
    batch_size: int = config.data.dataloader.batch_size
    max_length: int = config.data.dataloader.max_length
    is_truncation_random: bool = config.data.dataloader.is_truncation_random
    n_workers: int = config.data.dataloader.n_workers
    seed: int = config.main.seed

    train_loader = DataLoader(
        dataset=datasets['train'],
        batch_size=batch_size,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=datasets['val'],
        batch_size=batch_size,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=datasets['test'],
        batch_size=batch_size,
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