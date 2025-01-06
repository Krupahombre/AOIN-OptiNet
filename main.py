from src.data_loader import load_data
from src.utils import get_available_device

if __name__ == "__main__":
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    device = get_available_device()

    print(device)