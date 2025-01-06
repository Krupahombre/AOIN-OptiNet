import random

import torch

def get_available_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def generate_random_structure(min_layers=1, max_layers=3, min_neurons=10, max_neurons=100):
    num_layers = random.randint(min_layers, max_layers)
    structure = [random.randint(min_neurons, max_neurons) for _ in range(num_layers)]
    return structure


def print_population_info(population):
    print("Population Structure:")
    print("=" * 30)

    for i, individual in enumerate(population):
        layers_num, neurons_num = individual.get_structure_info()
        print(f"Individual {i + 1}:")
        print(f"  - Network structure: {layers_num} layers")
        print(f"  - Neurons in layers: {neurons_num}")
        print("-" * 30)
