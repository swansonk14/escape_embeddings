"""Contains a multilayer perceptron model class."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """A multilayer perceptron model."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layer_dims: tuple[int, ...]) -> None:
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the input to the model.
        :param hidden_layer_dims: The dimensionalities of the hidden layers.
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dims = hidden_layer_dims

        self.layer_dims = [self.input_dim] + list(self.hidden_layer_dims) + [self.output_dim]

        # Create layers
        self.layers = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            for i in range(len(self.layer_dims) - 1)
        ])

        # Create activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the model on the data.

        :param x: A FloatTensor containing an embedding of the antibody and/or antigen.
        :return: A FloatTensor containing the model's predicted escape score.
        """
        # Apply layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i != len(self.layers) - 1:
                x = self.activation(x)

        return x
