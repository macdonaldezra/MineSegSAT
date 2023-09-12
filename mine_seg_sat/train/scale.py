import json
from pathlib import Path
from statistics import mean

import torch

from mine_seg_sat.constants import BAND_ORDERING


def get_band_norm_values_from_root(root_dir: Path, min_max: bool = False):
    bands = {}
    global_max = 0
    band_paths = [
        e
        for e in list((root_dir / "metadata").glob("*_metadata.json"))
        if e.name.split("_metadata.json")[0] in BAND_ORDERING
    ]
    assert len(band_paths) > 0, f"No metadata found for {root_dir.as_posix()}"

    for band in band_paths:
        band_name = band.stem.split("_")[0]
        with open(band.as_posix(), "r") as f:
            metadata = json.load(f)
            if band_name in bands:
                if min_max:
                    bands[band_name] = (
                        min([float(bands[band_name][0]), float(metadata["min"])]),
                        max([float(bands[band_name][1]), float(metadata["max"])]),
                    )
                else:
                    bands[band_name] = (
                        mean(
                            [
                                float(bands[band_name][0]),
                                float(metadata["mean"]),
                            ]
                        ),
                        mean([float(bands[band_name][1]), float(metadata["std"])]),
                    )
            else:
                if min_max:
                    bands[band_name] = (
                        float(metadata["min"]),
                        float(metadata["max"]),
                    )
                else:
                    bands[band_name] = (
                        float(metadata["mean"]),
                        float(metadata["std"]),
                    )

    return (
        [bands[band][0] for band in BAND_ORDERING],
        [bands[band][1] for band in BAND_ORDERING],
        global_max,
    )


def get_labels(root_path: Path) -> dict[int, float]:
    """
    Given a root path, iterate over all directories in the root path,
    and return a dictionary containing the merged labels for each directory.
    """
    labels = {}
    for directory in root_path.iterdir():
        if not directory.is_dir():
            continue
        label_file = directory / "metadata" / "mask_metadata.json"
        with open(label_file, "r") as f:
            labels.update(json.load(f))

    return labels


def estimate_memory_inference(model, inputs, device: int = 0):
    """Predict the maximum memory usage of the model.
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be
            a single item, not a batch, and it will be replicated batch_size times.
        device (torch.device): the device to use
    """
    # Reset model and optimizer
    model.cpu()
    a = torch.cuda.memory_allocated(device)
    print(
        f"Memory available before loading model onto GPU: {format_bytes(torch.cuda.memory_allocated(device) - a)}"
    )

    model.to(device)
    print(
        f"Memory used by the model while loaded onto GPU: {format_bytes(torch.cuda.memory_allocated(device) - a)}"
    )
    inputs = inputs.to(device)
    _ = model(inputs)
    print(
        f"Memory used by model after computing output for an input on GPU: {format_bytes(torch.cuda.memory_allocated(device) - a)}"
    )


def format_bytes(bytes: int):
    gb = bytes / (1024**3)
    mb = (bytes % (1024**3)) / (1024**2)

    if gb >= 1:
        return f"{gb:.2f} GB"
    else:
        return f"{mb:.2f} MB"
