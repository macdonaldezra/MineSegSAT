import gc

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from mine_seg_sat.config import TrainingConfig, config_to_yaml, get_model_config
from mine_seg_sat.dataloader import get_dataloader
from mine_seg_sat.train_utils.binary import binary_segmentation_train
from mine_seg_sat.train_utils.utils import (
    generate_binary_test_metrics,
    get_loss,
    get_lr_scheduler,
    get_model,
)
from mine_seg_sat.utils.distributed import cleanup, setup
from mine_seg_sat.utils.path import get_experiment_outpath, load_ddp_model


def main(rank: int, world_size: int, config: TrainingConfig) -> None:
    """
    Train SegFormer on EUSegSatellite dataset
    """
    setup(rank, world_size)
    print(f"Rank: {rank} - World size: {world_size}")

    comment = f"dim={config.image_size}_epochs={config.epochs}_lr={config.learning_rate}_bs={config.batch_size}"
    if config.weight_filepath:
        comment += "_checkpoint"
    dist.barrier()  # Try and ensure that both systems use the same minute for the output path...
    config.output_path = get_experiment_outpath(
        config.output_path, config.model_name, comment=comment
    )
    if rank == 0:
        writer = SummaryWriter(config.output_path)
    else:
        writer = None

    device = torch.device("cuda", rank)
    dataloaders = get_dataloader(rank, world_size, config)
    dataset_len = len(dataloaders["train"].dataset) + len(dataloaders["val"].dataset)
    if rank == 0:
        print(
            f"Length of training data: {len(dataloaders['train'].dataset)}, "
            + f"Length of validation data: {len(dataloaders['val'].dataset)}"
        )
        config_to_yaml(config, config.output_path / "config.yaml")

    model = get_model(config, device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=config.learning_rate)

    criterion = get_loss(config, rank)
    scheduler = get_lr_scheduler(optimizer, config, dataset_len)

    if config.weight_filepath:
        if rank == 0:
            print(
                f"Loading pretrained weights from {config.weight_filepath.as_posix()}"
            )
        checkpoint = torch.load(config.weight_filepath.as_posix())
        current_model_dict = ddp_model.state_dict()
        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
            for k, v in zip(
                current_model_dict.keys(), checkpoint["model_state_dict"].values()
            )
        }
        ddp_model.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if config.num_classes == 1:
        binary_segmentation_train(
            config=config,
            dataloaders=dataloaders,
            rank=rank,
            optimizer=optimizer,
            model=ddp_model,
            criterion=criterion,
            lr_scheduler=scheduler,
            writer=writer,
        )
        if rank == 0:
            print(
                f"Loading best model from training at {config.output_path / 'best_model.pth'}..."
            )
        dist.barrier()  # Ensure all model saving events have finished before evaluating
        ddp_model = load_ddp_model(ddp_model, config.output_path / "best_model.pth")
        generate_binary_test_metrics(
            ddp_model, dataloaders["test"], device, config.num_classes
        )
    else:
        raise NotImplementedError(
            "Multiclass segmentation not implemented in this module."
        )

    gc.collect()
    torch.cuda.empty_cache()
    cleanup()


def run():
    config = get_model_config()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Requires at least 1 GPU to run, but got {n_gpus}"
    world_size = n_gpus

    mp.spawn(
        main,
        args=(
            world_size,
            config,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    run()
