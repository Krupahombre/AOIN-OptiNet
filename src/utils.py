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

def save_to_csv(file_name, max_fitness_count, generation, epochs, population_size, best_individual, selection_type, tournament_size, crossover_type, crossover_prob, mutation_type, mutation_prob):
    file_path = os.path.join(data_path, file_name)
    fieldnames = [
        "Max fitness Count", "Generation Number", "Epochs", "Population Size", "Selection Type", "Tournament Size", "Crossover Type", "Crossover Probability", "Mutation Type", "Mutation Probability", "Structure",
        "Number of Layers", "Input Layer", "Output Layer", "Train Time", "Accuracy"
    ]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()


    formatted_crossover_prob = round(crossover_prob, 3)
    formatted_mutation_prob = round(mutation_prob, 3)
    formatted_train_time = round(best_individual.get_train_time(), 4)
    formatted_accuracy = round(best_individual.get_accuracy(), 4)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            "Max fitness Count": 100,
            "Generation Number": generation,
            "Epochs": epochs,
            "Population Size": population_size,
            "Selection Type": selection_type,
            "Tournament Size": tournament_size,
            "Crossover Type": crossover_type,
            "Crossover Probability": formatted_crossover_prob,
            "Mutation Type": mutation_type,
            "Mutation Probability": formatted_mutation_prob,
            "Structure": best_individual.structure,
            "Number of Layers": len(best_individual.structure),
            "Input Layer": best_individual.input_layer,
            "Output Layer": best_individual.output_layer,
            "Train Time": best_individual.get_train_time(),
            "Accuracy": formatted_accuracy
        })