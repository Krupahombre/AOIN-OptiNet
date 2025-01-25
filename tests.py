from tests.test_selection import test_selection
from tests.test_mutation import test_mutation
from tests.test_crossover import test_crossover
from tests.test_population import test_population
from tests.test_epochs import test_epochs
from tests.test_crossover_probability import test_crossover_probability
from tests.test_mutation_probability import test_mutation_probability
from tests.test_tournament_size import test_tournament_size

if __name__ == "__main__":
    test_selection()
    test_crossover()
    test_mutation()
    test_population()
    test_epochs()
    test_crossover_probability()
    test_mutation_probability()
    test_tournament_size()
