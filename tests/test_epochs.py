import json
from src.evolutionary import run_evolutionary_optimization
from src.evolutionary_operator_loader import EvolutionaryOperatorLoader
from src.evolutionary_operator_manager import EvolutionaryOperatorManager
from src.utils import get_available_device, generate_random_structure
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data
import os

def test_epochs():
    csv_dir="epochs"

    with open('tests/init_config.json', 'r') as file:
        data = json.load(file)
    
    batch_size = data["batch_size"]
    population_size = data["population_size"]
    generations = data["generations"]
    epochs = data["epochs"]
    runs = data["runs"]
    selection_type = data["selection_type"]
    crossover_type = data["crossover_type"]
    mutation_type = data["mutation_type"]

    epochs_num = [1, 2, 3]


    os.makedirs(csv_dir, exist_ok=True)
    

    train_loader, test_loader = load_data(batch_size)
    device = get_available_device()
    manager = EvolutionaryOperatorManager()
    EvolutionaryOperatorLoader.load_operators(manager)

    for epochs in epochs_num:
        for i in range(runs):
            print(f'executing {i + 1} run for {epochs}')
            population = [
                CustomNeuralNetwork(generate_random_structure()) for _ in range(population_size)
            ]

            run_evolutionary_optimization(
                manager,
                population,
                train_loader,
                test_loader,
                epochs,
                device,
                generations,
                selection_type,
                crossover_type,
                mutation_type,
                csv_path=os.path.join(csv_dir, f"{selection_type}.csv")
            )
