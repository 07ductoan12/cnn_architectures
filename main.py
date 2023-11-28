import torch
from torch.utils.data import DataLoader
from custom_dataset import custom_dataset
from model.pytorch_vgg import vgg11
from model.resnet_pytorch import resnet18
from train_test_module import train
from trainner import Tranner


TRAIN_DIR = "../../datasets/flower/train/"
VALID_DIR = "../../datasets/flower/valid/"
NUM_WORKERS = 4
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.003


def load_dataset():
    train_dataset = custom_dataset(TRAIN_DIR, mode="Train")
    valid_dataset = custom_dataset(VALID_DIR, mode="Valid")

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return train_dataloader, valid_dataloader, len(train_dataset.classes)


def main():
    train_dataloader, valid_dataloader, num_classes = load_dataset()

    model = vgg11(num_classes=num_classes, batch_norm=False)
    # model = model = resnet18(num_classes=num_classes)
    model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=valid_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=DEVICE,
    )

    # trainner = Tranner(
    #     epochs=EPOCHS,
    #     device=DEVICE,
    #     disable_progress_bar=False,
    #     loss_fn=loss_fn,
    #     optimizer=optimizer,
    #     model=model,
    # )
    # trainner.train(train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
