from time import sleep
import datetime

from src.evolutionary_operator_manager import EvolutionaryOperatorManager
from src.neural_network_utils import train_network, test_network
from src.utils import save_to_csv


def evaluate(individual, train_loader, test_loader, device, epochs):
    train_time = train_network(individual, train_loader, device, epochs)
    test_loss, test_accuracy = test_network(individual, test_loader, device)

    individual.set_accuracy(test_accuracy)
    individual.set_train_time(round(train_time, 2))

    num_params = individual.get_num_parameters()
    return test_accuracy, train_time, num_params


def run_evolutionary_optimization(manager: EvolutionaryOperatorManager, max_fitness_count, population, train_loader, test_loader,
                                  epochs, device, generations,
                                  selection_type, crossover_type, mutation_type,
                                  crossover_prob, mutation_prob, tournament_size,
                                  csv_path="results"):
    selection_method = manager.get("selection", selection_type)
    crossover_method = manager.get("crossover", crossover_type)
    mutation_method = manager.get("mutation", mutation_type)
    csv_path = f"{csv_path}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")
        if max_fitness_count >= 0:
            for i, individual in enumerate(population):
                print(f"\n\tTraining individual {i + 1}/{len(population)}...")
                evaluate(individual, train_loader, test_loader, device, epochs)
                max_fitness_count -= 1
                sleep(0.2)

            population.sort(key=lambda x: x.get_accuracy(), reverse=True)

            best_individual = population[0]

            save_to_csv(csv_path, max_fitness_count, gen + 1, len(population), best_individual, selection_type, tournament_size, crossover_type,
                        crossover_prob, mutation_type, mutation_prob)

            best_individuals = population[:2]
            new_population = best_individuals[:]

            while len(new_population) < len(population):
                parent1 = selection_method(population, tournament_size)
                parent2 = selection_method(population, tournament_size)
                offspring = crossover_method(parent1, parent2, crossover_prob)
                for child in offspring:
                    mutation_method(child, mutation_prob)
                    new_population.append(child)

                    if len(new_population) >= len(population):
                        break

            population = new_population

    return max(population, key=lambda x: x.get_accuracy())