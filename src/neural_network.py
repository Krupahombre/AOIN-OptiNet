import torch.nn as nn

class CustomNeuralNetwork(nn.Module):
    def __init__(self, structure):
        super().__init__()
        self.flatten = nn.Flatten()
        self.structure = structure
        self.train_time = 0
        self.accuracy = 0
        self.input_layer = 28 * 28
        self.output_layer = 10
        layers = []

        for neurons in structure:
            layers.append(nn.Linear(self.input_layer, neurons))
            layers.append(nn.ReLU())
            self.input_layer = neurons

        layers.append(nn.Linear(self.input_layer, self.output_layer))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def get_accuracy(self):
        return self.accuracy

    def set_train_time(self, train_time):
        self.train_time = train_time

    def get_train_time(self):
        return self.train_time

    def get_structure_info(self):
        return len(self.structure), self.structure

    def get_layers_info(self):
        info = {
            "Input Layer": self.input_layer,
            "Hidden Layers": self.structure,
            "Output Layer": self.output_layer
        }
        total_layers = 1 + len(self.structure) + 1
        return {
            "Total Layers": total_layers,
            "Neurons per Layer": info
        }
