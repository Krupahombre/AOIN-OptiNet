from src.evolutionary import run_evolutionary_optimization
from src.evolutionary_operator_loader import EvolutionaryOperatorLoader
from src.evolutionary_operator_manager import EvolutionaryOperatorManager
from src.utils import get_available_device, generate_random_structure
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data
import os
from src.evolutionary_parameters_manager import EvolutionaryParametersManager

def test_mutation_probability():
    csv_dir="mutation_probability"

    params_manager = EvolutionaryParametersManager()

    batch_size = params_manager.get("batch_size")

    runs = params_manager.get("runs")

    mutation_probabilities = [0.05, 0.2, 0.5]

    os.makedirs(csv_dir, exist_ok=True)
    
    train_loader, test_loader = load_data(batch_size)
    device = get_available_device()
    manager = EvolutionaryOperatorManager()
    EvolutionaryOperatorLoader.load_operators(manager)

    for mutation_probability in mutation_probabilities:
        params_manager.set("mutation_prob", mutation_probability)
        for i in range(runs):
            print(f'executing {i + 1} run for {mutation_probability}')
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
                csv_path=os.path.join(csv_dir, f"{mutation_probability}.csv")
            )
