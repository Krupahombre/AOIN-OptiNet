import random

import torch

def get_available_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def generate_random_structure():
    num_layers = random.randint(min_layers, max_layers)
    structure = [random.randint(min_neurons, max_neurons) for _ in range(num_layers)]
    return structure