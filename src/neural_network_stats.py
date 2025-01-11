class NeuralNetworkStats:
    def __init__(self, structure, input_layer, output_layer, train_time=0, accuracy=0):
        self.structure = structure
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.train_time = train_time
        self.accuracy = accuracy

    def update(self, train_time=None, accuracy=None):
        if train_time is not None:
            self.train_time = train_time
        if accuracy is not None:
            self.accuracy = accuracy

    def get_summary(self):
        return {
            "Structure": self.structure,
            "Input layer": self.input_layer,
            "Output layer": self.output_layer,
            "Number of Layers": len(self.structure) + 2,
            "Train Time": self.train_time,
            "Accuracy": self.accuracy
        }