import random

from src.neural_network_utils import train_network, test_network


def evaluate(individual, train_loader, test_loader, device, epochs):
    train_time = train_network(individual, train_loader, device, epochs)
    test_loss, test_accuracy = test_network(individual, test_loader, device)

    individual.set_accuracy(test_accuracy)
    individual.set_train_time(round(train_time, 2))

    num_params = individual.get_num_parameters()
    return test_accuracy, train_time, num_params


def crossover(ind1, ind2):
    if random.random() < 0.5 and (len(ind1.structure) > 1 and len(ind2.structure) > 1):
        cxpoint = random.randint(1, len(ind1.structure) - 1)
        ind1.structure[cxpoint:], ind2.structure[cxpoint:] = ind2.structure[cxpoint:], ind1.structure[cxpoint:]
    return [ind1, ind2]


def mutate(individual):
    if random.random() < 0.2:
        layer_idx = random.randint(0, len(individual.structure) - 1)
        individual.structure[layer_idx] = random.randint(16, 128)
    return individual,


def tournament_selection(population, n=3):
    tournament = random.sample(population, n)
    tournament.sort(key=lambda x: x.get_accuracy(), reverse=True)
    return tournament[0]


def run_evolutionary_optimization(population, train_loader, test_loader, epochs, device, generations):
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")
        for i, individual in enumerate(population):
            print(f"  Training individual {i + 1}/{len(population)}...")
            evaluate(individual, train_loader, test_loader, device, epochs)

        population.sort(key=lambda x: x.get_accuracy(), reverse=True)

        best_individuals = population[:2]
        new_population = best_individuals[:]

        while len(new_population) < len(population):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            offspring = crossover(parent1, parent2)
            for child in offspring:
                mutate(child)
                new_population.append(child)

                if len(new_population) >= len(population):
                    break

        population = new_population

    return max(population, key=lambda x: x.get_accuracy())
