import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(
    epoch: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    disable_progress_bar: bool = False,
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar,
    )

    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
                "train_acc": train_acc / (batch + 1),
            }
        )

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    epoch: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    disable_progress_bar: bool = False,
) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar,
    )

    with torch.no_grad():
        for batch, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            progress_bar.set_postfix(
                {
                    "test_loss": test_loss / (batch + 1),
                    "test_acc": test_acc / (batch + 1),
                }
            )

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    disable_progress_bar: bool = False,
) -> Dict[str, List]:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "train_epoch_time": [],
        "test_epoch_time": [],
    }

    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        train_epoch_start_time = time.time()
        train_loss, train_acc = train_step(
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            disable_progress_bar=disable_progress_bar,
        )
        train_epoch_end_time = time.time()
        train_epoch_time = train_epoch_end_time - train_epoch_start_time

        test_epoch_start_time = time.time()
        test_loss, test_acc = test_step(
            epoch=epoch,
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            disable_progress_bar=disable_progress_bar,
        )
        test_epoch_end_time = time.time()
        test_epoch_time = test_epoch_end_time - test_epoch_start_time

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"train_epoch_time: {train_epoch_time:.4f} | "
            f"test_epoch_time: {test_epoch_time:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_epoch_time"].append(train_epoch_time)
        results["test_epoch_time"].append(test_epoch_time)

    return results
