import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Tuple


class Tranner:
    def __init__(
        self,
        epochs: int,
        device: torch.device,
        disable_progress_bar: bool,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
    ) -> None:
        self.epochs = epochs
        self.device = device
        self.disable_progress_bar = disable_progress_bar
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model

        self.results = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
            "train_epoch_time": [],
            "valid_epoch_time": [],
        }

    def train_step(self, epoch: int, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        train_loss, train_acc = 0, 0

        progress_bar = tqdm(
            enumerate(dataloader),
            desc=f"Training Epoch {epoch}",
            total=len(dataloader),
            disable=self.disable_progress_bar,
        )

        for batch, (X, y) in progress_bar:
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    def test_step(self, epoch: int, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        test_loss, test_acc = 0, 0

        progress_bar = tqdm(
            enumerate(dataloader),
            desc=f"Testing Epoch {epoch}",
            total=len(dataloader),
            disable=self.disable_progress_bar,
        )

        with torch.no_grad():
            for batch, (X, y) in progress_bar:
                X, y = X.to(self.device), y.to(self.device)

                test_pred_logits = self.model(X)

                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                test_pred_labels = torch.argmax(
                    torch.softmax(test_pred_logits, dim=1), dim=1
                )
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_logits)

                progress_bar.set_postfix(
                    {
                        "valid_loss": test_loss / (batch + 1),
                        "valid_acc": test_acc / (batch + 1),
                    }
                )

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    @staticmethod
    def cal_time(start_time, end_time):
        elapsed_time = end_time - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Time taken: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
        )
        return elapsed_time

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader):
        for epoch in tqdm(range(self.epochs), disable=self.disable_progress_bar):
            train_epoch_start_time = time.time()
            train_loss, train_acc = self.train_step(
                epoch=epoch, dataloader=train_dataloader
            )
            train_epoch_end_time = time.time()
            train_epoch_time = self.cal_time(
                train_epoch_start_time, train_epoch_end_time
            )

            valid_epoch_start_time = time.time()
            valid_loss, valid_acc = self.test_step(
                epoch=epoch, dataloader=valid_dataloader
            )
            valid_epoch_end_time = time.time()
            valid_epoch_time = self.cal_time(
                valid_epoch_start_time, valid_epoch_end_time
            )

            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"valid_loss: {valid_loss:.4f} | "
                f"valid_acc: {valid_acc:.4f} | "
                f"train_epoch_time: {train_epoch_time:.4f} | "
                f"valid_epoch_time: {valid_epoch_time:.4f}"
            )

            self.results["train_loss"].append(train_loss)
            self.results["train_acc"].append(train_acc)
            self.results["valid_loss"].append(valid_loss)
            self.results["valid_acc"].append(valid_acc)
            self.results["train_epoch_time"].append(train_epoch_time)
            self.results["valid_epoch_time"].append(valid_epoch_time)
