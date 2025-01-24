from tests.test_selection import test_selection
from tests.test_mutation import test_mutation
from tests.test_crossover import test_crossover
from tests.test_population import test_population
from tests.test_epochs import test_epochs

if __name__ == "__main__":
    test_selection()
    test_crossover()
    test_mutation()
    test_population()
    test_epochs()