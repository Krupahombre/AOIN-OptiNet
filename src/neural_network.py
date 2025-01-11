import torch.nn as nn

from src.neural_network_stats import NeuralNetworkStats


class CustomNeuralNetwork(nn.Module):
    def __init__(self, structure):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = 28 * 28
        input_layer_copy = self.input_layer
        self.output_layer = 10
        self.structure = structure

        layers = []
        for neurons in structure:
            layers.append(nn.Linear(input_layer_copy, neurons))
            layers.append(nn.ReLU())
            input_layer_copy = neurons

        layers.append(nn.Linear(input_layer_copy, self.output_layer))
        self.linear_relu_stack = nn.Sequential(*layers)

        self.stats = NeuralNetworkStats(structure, 28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_accuracy(self, accuracy):
        self.stats.update(accuracy=accuracy)

    def get_accuracy(self):
        return self.stats.accuracy

    def set_train_time(self, train_time):
        self.stats.update(train_time=train_time)

    def get_train_time(self):
        return self.stats.train_time

    def get_structure_info(self):
        return len(self.structure), self.structure

    def get_layers_info(self):
        return self.stats.get_summary()
