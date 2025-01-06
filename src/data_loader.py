import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

data_path = Path('./data')

def load_data(batch_size=64):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

    return train_loader, test_loader
