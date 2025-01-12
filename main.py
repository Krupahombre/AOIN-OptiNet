from src.evolutionary import run_evolutionary_optimization
from src.utils import get_available_device, generate_random_structure
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data


if __name__ == "__main__":
    batch_size = 64
    population_size = 3
    generations = 5
    epochs = 3
    train_loader, test_loader = load_data(batch_size)

    device = get_available_device()

    population = [
        CustomNeuralNetwork(generate_random_structure()) for _ in range(population_size)
    ]

    print("Starting Evolutionary Optimization...\n")

    best_individual = run_evolutionary_optimization(population, train_loader, test_loader, epochs, device, generations)
    print(f"\nBest Individual: {best_individual.stats.get_summary()}")
