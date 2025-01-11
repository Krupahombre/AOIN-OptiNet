from src.data_loader import load_data
from src.neural_network import CustomNeuralNetwork
from src.neural_network_utils import train_network, test_network
from src.utils import get_available_device, generate_random_structure, print_population_info

import random
import time
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import get_available_device
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data
from src.neural_network_utils import train_network, test_network


# --- Funkcja ewolucyjna ---
def evaluate(individual, train_loader, test_loader, device, epochs):
    """
    Ocena osobnika na podstawie dokładności, czasu trenowania i liczby parametrów.
    """
    train_time = train_network(individual, train_loader, device, epochs)
    test_loss, test_accuracy = test_network(individual, test_loader, device)

    individual.set_accuracy(test_accuracy)
    individual.set_train_time(round(train_time, 2))

    num_params = individual.get_num_parameters()
    return test_accuracy, train_time, num_params


# --- Operator krzyżowania ---
def crossover(ind1, ind2):
    """ Krzyżowanie dwóch osobników: wymiana warstw """
    if random.random() < 0.5 and (len(ind1.structure) > 1 and len(ind2.structure) > 1):
        cxpoint = random.randint(1, len(ind1.structure) - 1)
        ind1.structure[cxpoint:], ind2.structure[cxpoint:] = ind2.structure[cxpoint:], ind1.structure[cxpoint:]
    return ind1, ind2


# --- Operator mutacji ---
def mutate(individual):
    """ Mutacja: zmiana liczby neuronów w losowej warstwie """
    if random.random() < 0.2:
        layer_idx = random.randint(0, len(individual.structure) - 1)
        individual.structure[layer_idx] = random.randint(16, 128)
    return individual,


# --- Selekcja turniejowa ---
def tournament_selection(population, n=3):
    """ Selekcja turniejowa: wybieramy najlepszych na podstawie fitness """
    tournament = random.sample(population, n)
    tournament.sort(key=lambda x: x.get_accuracy(), reverse=True)
    return tournament[0]


# --- Główna funkcja ewolucyjna ---
def run_evolutionary_optimization(population, train_loader, test_loader, epochs, device, generations=10):
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")
        # Oceniamy wszystkich osobników
        for i, individual in enumerate(population):
            print(f"  Training individual {i + 1}/{len(population)}...")
            evaluate(individual, train_loader, test_loader, device, epochs)

        # Sortujemy populację według dokładności
        population.sort(key=lambda x: x.get_accuracy(), reverse=True)

        # Zbieramy najlepszych osobników
        best_individuals = population[:2]  # Wybieramy najlepszych 2 osobników

        # Krzyżowanie: tworzymy nowe osobniki
        new_population = best_individuals[:]

        while len(new_population) < len(population):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            offspring1, offspring2 = crossover(parent1, parent2)
            mutate(offspring1)
            mutate(offspring2)
            new_population.extend([offspring1, offspring2])

        population = new_population

    return population


# --- Główna część programu ---
if __name__ == "__main__":
    batch_size = 64
    population_size = 5
    epochs = 3
    train_loader, test_loader = load_data(batch_size)

    device = get_available_device()

    population = [
        CustomNeuralNetwork(generate_random_structure()) for _ in range(population_size)
    ]

    print("Starting Evolutionary Optimization...\n")
    trained_population = run_evolutionary_optimization(population, train_loader, test_loader, epochs, device)

    best_individual = max(trained_population, key=lambda x: x.get_accuracy())
    print(f"\nBest Individual: {best_individual.get_accuracy()}% Accuracy")

# def run_evolutionary_optimization(population, train_loader, test_loader, epochs, device):
#     for i, individual in enumerate(population):
#         print("-" * 50)
#         print(f"\n  Training individual {i + 1}/{len(population)}...")
#
#         layers_num, neurons_num = individual.get_structure_info()
#         train_time = train_network(individual, train_loader, device, epochs)
#         test_loss, test_accuracy = test_network(individual, test_loader, device)
#
#         individual.set_accuracy(test_accuracy)
#         individual.set_train_time(round(train_time, 2))
#
#         layers_info = individual.get_layers_info()
#
#         print(f"  Network structure: {layers_num} layers, neurons in layers: {neurons_num}")
#         print(f"  Layers info: {layers_info}")
#         print(f"  Training time: {train_time:.2f} seconds")
#         print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
#         print("-" * 50)
#
#     return population
#
#
# if __name__ == "__main__":
#     batch_size = 64
#     populations_size = 5
#     epochs = 3
#     train_loader, test_loader = load_data(batch_size)
#
#     device = get_available_device()
#
#     population = [
#         CustomNeuralNetwork(generate_random_structure()) for _ in range(populations_size)
#     ]
#
#     print_population_info(population)
#
#     trained_population = run_evolutionary_optimization(population, train_loader, test_loader, epochs, device)
