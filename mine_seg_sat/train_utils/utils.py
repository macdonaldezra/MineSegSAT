import typing as T
from pathlib import Path

import pandas as pd
import torch
from segmentation_models_pytorch import Unet, UnetPlusPlus
from segmentation_models_pytorch.losses import (
    DiceLoss,
    FocalLoss,
    LovaszLoss,
    TverskyLoss,
)
from torchmetrics import (
    Accuracy,
    ClasswiseWrapper,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)

from mine_seg_sat.config import TrainingConfig
from mine_seg_sat.models.segformer import SegFormer


def get_lr_scheduler(
    optimizer: torch.nn.Module, config: TrainingConfig, dataset_len: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Retrieve a learning rate scheduler for the given optimizer.
    """
    if config.lr_scheduler == "cosine_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=config.min_learning_rate, verbose=False
        )
    if config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_learning_rate,
            verbose=True,
        )
    elif config.lr_scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_learning_rate,
            epochs=config.epochs,
            steps_per_epoch=dataset_len,
            pct_start=0.3,
            div_factor=2,
            final_div_factor=5,
            anneal_strategy="cos",
            verbose=False,
        )
    elif config.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,
            patience=8,
            min_lr=config.min_learning_rate,
            verbose=True,
        )
    elif config.lr_scheduler == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=config.epochs, power=2, last_epoch=config.epochs * 2
        )
    else:
        raise ValueError(f"Learning rate not found for {config.lr_scheduler}.")


def get_loss(config: TrainingConfig, rank: T.Optional[str] = None) -> torch.nn.Module:
    """
    Retrieve a loss function for the given configuration.
    """
    if config.num_classes == 1:
        mode = "binary"
    else:
        mode = "multiclass"
    if config.loss == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif config.loss == "dice":
        return DiceLoss(
            mode,
            smooth=1.0,
            from_logits=True,
            ignore_index=config.ignore_index,
            classes=config.included_classes,
        )
    elif config.loss == "focal":
        return FocalLoss(mode, alpha=2)
    elif config.loss == "tversky":
        return TverskyLoss(mode, smooth=1, alpha=4, beta=4)
    elif config.loss == "lovasz":
        return LovaszLoss(mode, ignore_index=config.ignore_index)
    else:
        raise ValueError(f"Loss not found for {config.loss}.")


def get_model(config: TrainingConfig, device: torch.device) -> torch.nn.Module:
    """
    Get a model for training.
    """
    if config.model == "unet":
        model = Unet(
            encoder_name=config.encoder,
            decoder_attention_type="scse",
            encoder_weights=None,  # can't use weights with 3-channel input...
            classes=config.num_classes,
            in_channels=config.in_channels,
        )
    elif config.model == "unetplusplus":
        model = UnetPlusPlus(
            encoder_name=config.encoder,
            decoder_attention_type="scse",
            encoder_weights=None,
            classes=config.num_classes,
            in_channels=config.in_channels,
        )
    elif config.model == "segformer":
        model = SegFormer(
            in_channels=config.in_channels, num_classes=config.num_classes
        )
    else:
        raise ValueError(f"Model not found for {config.model}.")

    return model.to(device)


def get_metrics(
    num_classes: int, device: str, labels: T.Optional[T.List[str]] = None
) -> T.Dict[str, torch.nn.Module]:
    """
    Get metrics for model.
    """
    if num_classes > 1 and labels is None:
        raise ValueError("Labels must be provided for multiclass metrics.")

    if num_classes < 2:
        return MetricCollection(
            {
                "F1": F1Score(
                    task="binary", average="micro", num_classes=num_classes
                ).to(device),
                "Precision": Precision(
                    task="binary", average="micro", num_classes=num_classes
                ).to(device),
                "Recall": Recall(
                    task="binary", average="micro", num_classes=num_classes
                ).to(device),
                "Jaccard": JaccardIndex(
                    task="binary", average="micro", num_classes=num_classes
                ).to(device),
                "Accuracy": Accuracy(
                    task="binary", average="micro", num_classes=num_classes
                ).to(device),
            }
        )

    return MetricCollection(
        {
            "F1": ClasswiseWrapper(
                F1Score(task="multiclass", average=None, num_classes=num_classes),
                labels,
            ).to(device),
            "Precision": ClasswiseWrapper(
                Precision(task="multiclass", average=None, num_classes=num_classes),
                labels,
            ).to(device),
            "Recall": ClasswiseWrapper(
                Recall(task="multiclass", average=None, num_classes=num_classes), labels
            ).to(device),
            "Jaccard": ClasswiseWrapper(
                JaccardIndex(task="multiclass", average=None, num_classes=num_classes),
                labels,
            ).to(device),
            "Accuracy": ClasswiseWrapper(
                Accuracy(task="multiclass", average=None, num_classes=num_classes),
                labels,
            ).to(device),
        }
    )


def get_metric_df(metrics: T.Dict[str, T.Dict[str, float]]) -> pd.DataFrame:
    """
    Get metric dataframe.
    """
    columns = ["F1", "Accuracy", "Precision", "Recall", "Jaccard"]
    indices = [
        name.split("_")[1]
        for name in metrics.keys()
        if columns[0].casefold() in name.casefold()
    ]

    metric_df = pd.DataFrame(columns=columns, index=indices)
    for metric, value in metrics.items():
        for column in columns:
            if column.casefold() in metric.casefold():
                metric_df.loc[metric.split("_")[1], column] = value.cpu().item()

    return metric_df


def generate_binary_test_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> T.Dict[str, T.Dict[str, float]]:
    """
    Generate test metrics.
    """
    metrics = get_metrics(num_classes, device)
    model.eval()

    if "0" in str(device):
        print("Computing output metrics on test dataset...")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            prob_mask = outputs.sigmoid().float()
            labels = labels.unsqueeze(1)
            metrics.update(prob_mask, labels)

    out_metrics = metrics.compute()
    metrics.reset()

    if "0" in str(device):
        out_string = "Final Epoch Test "
        for entry in out_metrics.keys():
            out_string += f" {entry}: {out_metrics[entry].cpu().item()}"

        print(out_string)


def generate_test_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    output_path: Path,
) -> T.Dict[str, T.Dict[str, float]]:
    """
    Generate test metrics.
    """
    if num_classes < 2:
        metrics = get_metrics(num_classes, device)
    else:
        metrics = get_metrics(
            num_classes, device, list(dataloader.dataset.index_to_label.values())
        )

    model.eval()

    if "0" in device:
        print("Computing output metrics on test dataset...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            images, masks = inputs.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1).long()
            metrics.update(outputs, masks)

    out_metrics = metrics.compute()
    metrics.reset()

    if "0" in device:
        if num_classes > 2:
            metric_df = get_metric_df(out_metrics)
            print(
                f"Number of rows that have accuracy greater than 0: {len(metric_df.loc[metric_df.Accuracy > 0])}"
            )
            print(f"Saving metrics to {output_path / 'test_metrics.csv'}")
            metric_df.to_csv(output_path / "test_metrics.csv")
            print(f"Head of metrics:\n{metric_df.loc[metric_df.Accuracy > 0].head(30)}")
        else:
            for entry in out_metrics.keys():
                print(f"{entry}: {out_metrics[entry].cpu().item()}")
