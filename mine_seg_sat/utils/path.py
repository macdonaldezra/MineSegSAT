import calendar
import datetime
import random
import string
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from mine_seg_sat.config import TrainingConfig


def get_checkpoint_path(out_path: Path, model_type: str, model_directory: str) -> Path:
    """
    Return a path to the checkpoint directory.
    """
    checkpoint_path = out_path / model_type / "checkpoints" / model_directory
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    return checkpoint_path


def get_output_path(out_path: Path, model_type: str, model_directory: str) -> Path:
    out_path = out_path / model_type / "out" / model_directory
    out_path.mkdir(exist_ok=True, parents=True)

    return out_path


def generate_random_string(length):
    characters = string.ascii_letters + string.digits  # Letters (both cases) and digits
    result = "".join(random.choice(characters) for _ in range(length))

    return result


def get_experiment_outpath(
    out_path: Path, model_name: str, comment: Optional[str]
) -> Path:
    today = datetime.datetime.now()
    month_date = calendar.month_abbr[today.month].lower() + str(today.day)
    training_init_time = today.strftime("%H%M%S")
    out_path = out_path / model_name / month_date / training_init_time
    if comment:
        out_path = out_path / comment

    out_path.mkdir(exist_ok=True, parents=True)

    return out_path


def add_module_prefix_if_missing(state_dict):
    """Prepends 'module.' to keys in the state_dict if it's not already there."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("module."):
            new_key = f"module.{k}"
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict


def load_ddp_model(model: nn.Module, path: Path) -> nn.Module:
    """
    Load a model from the given path and wrap it in Distributed Data Parallel (DDP).
    """
    device = torch.device("cuda", torch.distributed.get_rank())
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {path.as_posix()}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = add_module_prefix_if_missing(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)

    return model


def save_model(
    config: TrainingConfig,
    epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    model_name: str = "best_model.pth",
) -> None:
    """
    Function to save the trained model to disk.
    """
    print(f"Saving model after epoch {epochs} at: {config.output_path.as_posix()}...")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        (config.output_path / model_name).as_posix(),
    )
