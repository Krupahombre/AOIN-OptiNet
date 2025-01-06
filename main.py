from src.data_loader import load_data
from src.neural_network import CustomNeuralNetwork
from src.neural_network_utils import train_network, test_network
from src.utils import get_available_device, generate_random_structure, print_population_info


def run_evolutionary_optimization(population, train_loader, test_loader, epochs, device):
    for i, individual in enumerate(population):
        print(f"\nTraining individual {i + 1}/{len(population)}...")

        layers_num, neurons_num = individual.get_structure_info()
        train_time = train_network(individual, train_loader, device, epochs)
        test_loss, test_accuracy = test_network(individual, test_loader, device)

        print(f"  Network structure: {layers_num} layers, neurons in layers: {neurons_num}")
        print(f"  Training time: {train_time:.2f} seconds")
        print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    batch_size = 64
    populations_size = 5
    epochs = 3
    train_loader, test_loader = load_data(batch_size)

    device = get_available_device()

    population = [
        CustomNeuralNetwork(generate_random_structure()) for _ in range(populations_size)
    ]

    print_population_info(population)

    run_evolutionary_optimization(population, train_loader, test_loader, epochs, device)
