from src.evolutionary import run_evolutionary_optimization
from src.utils import get_available_device, generate_random_structure
from src.neural_network import CustomNeuralNetwork
from src.data_loader import load_data


def test_mutation_type(mutation_type, population, train_loader, test_loader, epochs, device, generations):
    print(f"\n--- Testing Mutation Type: {mutation_type} ---\n")
    best_individual = run_evolutionary_optimization(
        population=population,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        device=device,
        generations=generations,
        mutation_type=mutation_type
    )
    print(f"\nBest Individual with {mutation_type} mutation: {best_individual.stats.get_summary()}\n")
    return best_individual


if __name__ == "__main__":
    batch_size = 64
    population_size = 2
    generations = 2
    epochs = 2

    train_loader, test_loader = load_data(batch_size)
    device = get_available_device()

    population = [
        CustomNeuralNetwork(generate_random_structure()) for _ in range(population_size)
    ]

    mutation_types = ["layer_size", "structure_change", "add_layer"]
    results = []

    print("Starting Mutation Type Tests...\n")

    for mutation_type in mutation_types:
        test_population = [
            CustomNeuralNetwork(individual.structure[:]) for individual in population
        ]
        best_individual = test_mutation_type(mutation_type, test_population, train_loader, test_loader, epochs, device, generations)
        results.append((mutation_type, best_individual.get_accuracy()))

    print("\n--- Summary of Results ---")
    for mutation_type, accuracy in results:
        print(f"Mutation Type: {mutation_type}, Best Accuracy: {accuracy:.2f}")
