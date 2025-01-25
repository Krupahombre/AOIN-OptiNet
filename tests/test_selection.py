import os
from src.evolutionary import run_evolutionary_optimization
from src.evolutionary_operator_loader import EvolutionaryOperatorLoader
from src.evolutionary_operator_manager import EvolutionaryOperatorManager
from src.utils import get_available_device, generate_random_structure
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data
from src.evolutionary_parameters_manager import EvolutionaryParametersManager

def test_selection():
    csv_dir="selection"

    params_manager = EvolutionaryParametersManager()

    batch_size = params_manager.get("batch_size")

    runs = params_manager.get("runs")

    selection_types = ["tournament", "roulette"]

    os.makedirs(csv_dir, exist_ok=True)

    train_loader, test_loader = load_data(batch_size)
    device = get_available_device()
    manager = EvolutionaryOperatorManager()
    EvolutionaryOperatorLoader.load_operators(manager)

    for selection_type in selection_types:
        params_manager.set("selection_type", selection_type)
        for i in range(runs):
            print(f'executing {i + 1} run for {selection_type}')
            population = [
                CustomNeuralNetwork(generate_random_structure()) for _ in range(params_manager.get("population_size"))
            ]

            run_evolutionary_optimization(
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
                params_manager.get("tournament_size"),
                csv_path=os.path.join(csv_dir, f"{selection_type}.csv")
            )
