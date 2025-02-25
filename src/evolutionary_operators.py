import random


# SELECTION
def tournament_selection(population, n):
    if n > len(population):
        n = len(population)
    tournament = random.sample(population, n)
    tournament.sort(key=lambda x: x.get_accuracy(), reverse=True)
    return tournament[0]


def roulette_selection(population, size):
    total_fitness = sum(ind.get_accuracy() for ind in population)
    pick = random.uniform(0, total_fitness)
    current = 0

    for ind in population:
        current += ind.get_accuracy()
        if current >= pick:
            return ind


# CROSSOVER
def single_point_crossover(ind1, ind2, crossover_prob):
    if random.random() < crossover_prob and (len(ind1.structure) > 1 and len(ind2.structure) > 1):
        cxpoint = random.randint(1, len(ind1.structure) - 1)
        ind1.structure[cxpoint:], ind2.structure[cxpoint:] = ind2.structure[cxpoint:], ind1.structure[cxpoint:]
    return [ind1, ind2]


def unified_crossover(ind1, ind2, crossover_prob):
    alpha = 0.5
    if len(ind1.structure) != len(ind2.structure) or random.random() < crossover_prob:
        return [ind1, ind2]

    child1_structure = []
    child2_structure = []

    for val1, val2 in zip(ind1.structure, ind2.structure):
        if random.random() < alpha:
            child1_structure.append(val1)
            child2_structure.append(val2)
        else:
            child1_structure.append(val2)
            child2_structure.append(val1)

    ind1.structure = child1_structure
    ind2.structure = child2_structure

    return [ind1, ind2]


# MUTATION
def neuron_change_mutation(ind, mutation_prob):
    if random.random() < mutation_prob:
        layer_idx = random.randint(0, len(ind.structure) - 1)
        ind.structure[layer_idx] = random.randint(16, 128)

    return ind


def structure_change_mutation(ind, mutation_prob):
    if random.random() < mutation_prob:
        if random.random() < 0.5 and len(ind.structure) > 1:
            del ind.structure[random.randint(0, len(ind.structure) - 1)]
        else:
            ind.structure.append(random.randint(16, 128))

    return ind
