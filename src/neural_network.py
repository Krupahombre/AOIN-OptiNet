import torch.nn as nn

class CustomNeuralNetwork(nn.Module):
    def __init__(self, structure):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        input_size = 28 * 28

        for neurons in structure:
            layers.append(nn.Linear(input_size, neurons))
            layers.append(nn.ReLU())
            input_size = neurons

        layers.append(nn.Linear(input_size, 10))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits