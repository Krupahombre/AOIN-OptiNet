import random
import csv
import os
from src import data_path

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

def save_to_csv(file_path, generation, best_individual, mutation_type):
    file_path = os.path.join(data_path, file_path)
    fieldnames = [
        "Generation", "Mutation Type", "Structure", "Number of Layers",
        "Input layer", "Output layer", "Train Time", "Accuracy"
    ]

    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            "Generation": generation,
            "Mutation Type": mutation_type,
            "Structure": best_individual.structure,
            "Number of Layers": len(best_individual.structure),
            "Input layer": best_individual.input_layer,
            "Output layer": best_individual.output_layer,
            "Train Time": best_individual.get_train_time(),
            "Accuracy": best_individual.get_accuracy()
        })