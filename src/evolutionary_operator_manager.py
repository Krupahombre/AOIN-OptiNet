class EvolutionaryOperatorManager:
    def __init__(self) -> None:
        self.operators = {}

    def register(self, category, name, function) -> None:
        if category not in self.operators:
            self.operators[category] = {}

        self.operators[category][name] = function

    def get(self, category, name) -> callable:
        if category not in self.operators or name not in self.operators[category]:
            raise ValueError(f"Unknown method '{name}' for category '{category}'")

        return self.operators[category][name]

    def list_operators(self, category) -> list:
        if category not in self.operators:
            return []

        return list(self.operators[category].keys())
