import torch.nn as nn

def create_network(structure):
    layers = []
    input_size = 28 * 28

    for neurons in structure:
        layers.append(nn.Linear(input_size, neurons))
        layers.append(nn.ReLU())
        input_size = neurons

    layers.append(nn.Linear(input_size, 10))
    return nn.Sequential(*layers)