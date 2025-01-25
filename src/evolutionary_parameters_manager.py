class EvolutionaryParametersManager:
    def __init__(self, **kwargs):
        self.parameters = {
            "batch_size": 64,
            "population_size": 10,
            "generations": 5,
            "epochs": 1,
            "crossover_prob": 0.5,
            "mutation_prob": 0.2,
            "tournament_size": 3,
            "max_fitness_count": 100,
            "selection_type": "tournament",
            "crossover_type": "single_point",
            "mutation_type": "structure_change",
            "runs": 3
        }

        self.parameters.update(kwargs)

    def get(self, key, default=None):
        return self.parameters.get(key, default)

    def set(self, key, value):
        self.parameters[key] = value

    def all(self):
        return self.parameters
