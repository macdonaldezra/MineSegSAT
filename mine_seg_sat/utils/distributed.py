import os

import torch


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
