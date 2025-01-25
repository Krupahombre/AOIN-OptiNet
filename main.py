from src.evolutionary import run_evolutionary_optimization
from src.evolutionary_operator_loader import EvolutionaryOperatorLoader
from src.evolutionary_operator_manager import EvolutionaryOperatorManager
from src.evolutionary_parameters_manager import EvolutionaryParametersManager
from src.utils import get_available_device, generate_random_structure
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data


if __name__ == "__main__":
    params_manager = EvolutionaryParametersManager(
        population_size=5,
        generations=10,
        crossover_prob=0.5,
        mutation_prob=0.5
    )

    train_loader, test_loader = load_data(params_manager.get("batch_size"))

    device = get_available_device()
    manager = EvolutionaryOperatorManager()
    EvolutionaryOperatorLoader.load_operators(manager)

    population = [
        CustomNeuralNetwork(generate_random_structure()) for _ in range(params_manager.get("population_size"))
    ]

    print("Starting Evolutionary Optimization...\n")

    best_individual = run_evolutionary_optimization(
        manager,
        params_manager.get("max_fitness_count"),
        population,
        train_loader,
        test_loader,
        params_manager.get("epochs"),
        device,
        params_manager.get("generations"),
        params_manager.get("selection_type"),
        params_manager.get("crossover_type"),
        params_manager.get("mutation_type"),
        params_manager.get("crossover_prob"),
        params_manager.get("mutation_prob"),
        params_manager.get("tournament_size")
    )
    print(f"\nBest Individual: {best_individual.stats.get_summary()}")