#!/usr/bin/env python3

# Third party imports
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multilayer perceptron model."""
    def __init__(self, in_features: int, hidden_neurons: int):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_features, out_features=hidden_neurons
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_neurons, out_features=in_features
        )

    def forward(self, features):
        hidden_z = self.encoder_hidden_layer(features)
        hidden_y = torch.relu(hidden_z)
        output_z = self.decoder_output_layer(hidden_y)
        output_y = torch.relu(output_z)
        return output_y
