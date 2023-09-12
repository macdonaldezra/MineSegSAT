import typing as T

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, MeanMetric, Precision, Recall

from mine_seg_sat.config import TrainingConfig
from mine_seg_sat.utils.path import save_model


def binary_segmentation_train(
    config: TrainingConfig,
    rank: int,
    model: torch.nn.Module,
    dataloaders: dict[torch.nn.Module],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    writer: T.Optional[SummaryWriter] = None,
) -> None:
    # Train a binary image segmentation model
    device = torch.device("cuda", rank)
    accuracy = Accuracy(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    epoch_loss = MeanMetric().to(device)
    best_f1 = 0.0

    for epoch in range(config.epochs):
        dataloaders["train"].sampler.set_epoch(epoch)
        dataloaders["val"].sampler.set_epoch(epoch)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # compute metrics
                    prob_mask = outputs.sigmoid().float()
                    labels = labels.unsqueeze(1)

                    accuracy.update(prob_mask, labels)
                    f1.update(prob_mask, labels)
                    recall.update(prob_mask, labels)
                    precision.update(prob_mask, labels)
                    epoch_loss.update(loss)

            # Each metric needs to be computed in both phases
            acc = accuracy.compute()
            f1_score = f1.compute()
            prec = precision.compute()
            rec = recall.compute()
            ep_loss = epoch_loss.compute()

            if rank == 0:
                print(
                    f"Epoch {epoch+1} {phase.capitalize()} Accuracy: {acc}, F1: {f1_score}, Precision: {prec}, Recall: {rec}, Loss: {ep_loss.item()}"
                )
                if writer is not None:
                    writer.add_scalar(f"{phase}/Accuracy", acc, epoch)
                    writer.add_scalar(f"{phase}/F1", f1_score, epoch)
                    writer.add_scalar(f"{phase}/Precision", prec, epoch)
                    writer.add_scalar(f"{phase}/Recall", rec, epoch)
                    writer.add_scalar(f"{phase}/Loss", ep_loss, epoch)

            if phase == "val":
                if f1_score.item() > best_f1:
                    best_f1 = f1_score.item()
                    if rank == 0:
                        save_model(
                            config,
                            epoch,
                            model,
                            optimizer,
                            criterion,
                        )
                if config.lr_scheduler == "plateau":
                    dist.barrier()
                    lr_scheduler.step(ep_loss)
                if (
                    config.lr_scheduler == "cosine"
                    or config.lr_scheduler == "polynomial"
                ):
                    dist.barrier()
                    lr_scheduler.step()

            # reset metrics before next phase epoch
            accuracy.reset()
            epoch_loss.reset()
            f1.reset()
            precision.reset()
            recall.reset()

    if rank == 0:
        save_model(
            config,
            config.epochs,
            model,
            optimizer,
            criterion,
            model_name="final_model.pth",
        )
