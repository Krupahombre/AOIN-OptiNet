from src.evolutionary import run_evolutionary_optimization
from src.evolutionary_operator_loader import EvolutionaryOperatorLoader
from src.evolutionary_operator_manager import EvolutionaryOperatorManager
from src.utils import get_available_device, generate_random_structure
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data


if __name__ == "__main__":
    batch_size = 64
    population_size = 3
    generations = 5
    epochs = 1
    selection_type = "tournament"
    crossover_type = "single_point"
    mutation_type = "structure_change"
    train_loader, test_loader = load_data(batch_size)

    device = get_available_device()
    manager = EvolutionaryOperatorManager()
    EvolutionaryOperatorLoader.load_operators(manager)

    population = [
        CustomNeuralNetwork(generate_random_structure()) for _ in range(population_size)
    ]

    print("Starting Evolutionary Optimization...\n")

    best_individual = run_evolutionary_optimization(
        manager,
        population,
        train_loader,
        test_loader,
        epochs,
        device,
        generations,
        selection_type,
        crossover_type,
        mutation_type
    )
    print(f"\nBest Individual: {best_individual.stats.get_summary()}")