import time

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def train_network(network, train_loader, device, epochs):
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(epochs):
        network.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{epochs}") as batch_progress:
            for batch, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                predictions = network(images)
                loss = loss_fn(predictions, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(predictions, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                batch_progress.set_postfix(loss=running_loss / (batch + 1), accuracy=(correct_predictions / total_samples) * 100)

    end_time = time.time()
    training_duration = end_time - start_time

    return training_duration

def test_network(network, test_loader, device):
    network.to(device)
    network.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", desc="Testing") as batch_progress:
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                predictions = network(images)
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(predictions, labels)
                running_loss += loss.item()

                _, predicted = torch.max(predictions, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                batch_progress.set_postfix(loss=running_loss / (total_samples / len(test_loader)), accuracy=(correct_predictions / total_samples) * 100)

    avg_loss = running_loss / len(test_loader)
    accuracy = (correct_predictions / total_samples) * 100

    return avg_loss, accuracy
