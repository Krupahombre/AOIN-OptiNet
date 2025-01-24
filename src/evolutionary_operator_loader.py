from src.evolutionary_operator_manager import EvolutionaryOperatorManager
from src.evolutionary_operators import (
    tournament_selection,
    roulette_selection,
    single_point_crossover,
    unified_crossover,
    neuron_change_mutation,
    structure_change_mutation
)


class EvolutionaryOperatorLoader:
    @staticmethod
    def load_operators(manager: EvolutionaryOperatorManager):
        manager.register("selection", "tournament", tournament_selection)
        manager.register("selection", "roulette", roulette_selection)

        manager.register("crossover", "single_point", single_point_crossover)
        manager.register("crossover", "unified", unified_crossover)

        manager.register("mutation", "neuron_change", neuron_change_mutation)
        manager.register("mutation", "structure_change", structure_change_mutation)
