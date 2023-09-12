import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mine_seg_sat.config import TrainingConfig
from mine_seg_sat.dataset import MineSATDataset
from mine_seg_sat.train_utils.scale import get_band_norm_values_from_root
from mine_seg_sat.train_utils.transforms import (
    get_test_transforms,
    get_training_transforms,
    get_validation_transforms,
)


def get_minesat_dataloaders(
    rank: int, world_size: int, config: TrainingConfig
) -> dict[str, DataLoader]:
    """
    Return training and validation dataloaders for EuroSegSat dataset.
    """
    band_meta = get_band_norm_values_from_root(config.data_path, min_max=True)
    train_transforms = get_training_transforms(config.image_size)
    val_transforms = get_validation_transforms(config.image_size)
    test_transforms = get_test_transforms(config.image_size)

    train_dataset = MineSATDataset(
        split="train",
        data_path=config.data_path,
        transformations=train_transforms,
        min_max_normalization=True,
        max_values=band_meta[1],
    )
    val_dataset = MineSATDataset(
        split="val",
        data_path=config.data_path,
        transformations=val_transforms,
        min_max_normalization=True,
        max_values=band_meta[1],
    )
    test_dataset = MineSATDataset(
        split="test",
        data_path=config.data_path,
        transformations=test_transforms,
        min_max_normalization=True,
        max_values=band_meta[1],
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_dataloader_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_dataloader_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_dataloader_workers,
        pin_memory=False,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_dataloader(rank: int, world_size: int, config: TrainingConfig) -> dict:
    if config.dataset == "minesegsat":
        return get_minesat_dataloaders(rank, world_size, config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")
