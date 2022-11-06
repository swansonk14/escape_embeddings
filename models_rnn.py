"""Contains RNN model classes for predicting escape."""
from typing import Optional

import torch
import torch.nn as nn


from constants import (
    AA_TO_INDEX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_HIDDEN_LAYER_DIMS,
    EMBEDDING_GRANULARITY_OPTIONS,
    RBD_AA_INDICES,
    RBD_START_SITE,
    TASK_TYPE_OPTIONS
)
from models_mlp import MLP
from models_pytorch import PyTorchEscapeModel


class RNNCoreModel(nn.Module):
    """The core recurrent neural network that predicts escape scores based on one-hot amino acid features."""

    def __init__(self,
                 binarize: bool,
                 antigen_embedding_granularity: EMBEDDING_GRANULARITY_OPTIONS,
                 hidden_dim: int,
                 hidden_layer_dims: tuple[int, ...]) -> None:
        """Initialize the model.

        :param binarize: Whether the escape scores are binarized (for classification), thus requiring a sigmoid.
        :param antigen_embedding_granularity: The granularity of the antigen embeddings, either a sequence average or per-residue embeddings.
        :param hidden_dim: The dimension of the hidden state of the RNN.
        :param hidden_layer_dims: The dimensions of the hidden layers of the MLP.
        """
        super(RNNCoreModel, self).__init__()

        self.binarize = binarize
        self.antigen_embedding_granularity = antigen_embedding_granularity
        self.hidden_dim = hidden_dim
        self.hidden_layer_dims = hidden_layer_dims

        self.output_dim = 1

        # Create amino acid embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=len(AA_TO_INDEX),
            embedding_dim=self.hidden_dim
        )

        # Create RNN model
        self.rnn = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True
        )

        # Create MLP model
        self.mlp = MLP(
            input_dim=self.hidden_dim * (1 + self.rnn.bidirectional),
            output_dim=self.output_dim,
            hidden_layer_dims=self.hidden_layer_dims
        )

        # Create sigmoid function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the model on the data.

        :param x: A FloatTensor containing an embedding of the antibody and/or antigen.
        :return: A FloatTensor containing the model's predicted escape score.
        """
        # Extract mutation sites
        site_indices, x = x[0], x[1:]

        # Get batch size
        batch_size = x.shape[1]

        # Embed amino acids
        x = self.embeddings(x)

        # Run RNN (output: (sequence_length, batch_size, 2 * hidden_dim), hidden/cell: (2, batch_size, hidden_dim))
        output, (hidden, cell) = self.rnn(x)

        # Get sequence or residue embedding
        if self.antigen_embedding_granularity == 'sequence':
            x = hidden.transpose(0, 1).reshape(batch_size, -1)  # (batch_size, 2 * hidden_dim)
        elif self.antigen_embedding_granularity == 'residue':
            x = output[site_indices, torch.arange(batch_size)]  # (batch_size, 2 * hidden_dim)
        else:
            raise ValueError(f'Antigen embedding granularity "{self.antigen_embedding_granularity}" is not supported.')

        # Apply MLP
        x = self.mlp(x)

        # Apply sigmoid if appropriate
        if not self.training and self.binarize:
            x = self.sigmoid(x)

        # Squeeze output dimension
        x = x.squeeze(dim=1)

        return x


class RNNModel(PyTorchEscapeModel):
    """A recurrent neural network that predicts escape scores based on one-hot amino acid features."""

    def __init__(self,
                 task_type: TASK_TYPE_OPTIONS,
                 antigen_embedding_granularity: EMBEDDING_GRANULARITY_OPTIONS,
                 num_epochs: int,
                 hidden_dim: int,
                 hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str = DEFAULT_DEVICE) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        :param antigen_embedding_granularity: The granularity of the antigen embeddings, either a sequence average or per-residue embeddings.
        :param num_epochs: The number of epochs to train for.
        :param hidden_dim: The dimension of the hidden state of the RNN.
        :param hidden_layer_dims: The dimensions of the hidden layers of the MLP.
        :param batch_size: The number of sequences to process at once.
        :param device: The device to use (e.g., "cpu" or "cuda") for the RNN and embedding models.
        """
        super(RNNModel, self).__init__(
            task_type=task_type,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        self.antigen_embedding_granularity = antigen_embedding_granularity
        self.hidden_dim = hidden_dim
        self.hidden_layer_dims = hidden_layer_dims

        # Create model
        self._core_model = RNNCoreModel(
            binarize=self.binarize,
            antigen_embedding_granularity=self.antigen_embedding_granularity,
            hidden_dim=self.hidden_dim,
            hidden_layer_dims=self.hidden_layer_dims
        ).to(self.device)

        # Create optimizer
        self._optimizer = torch.optim.Adam(self.core_model.parameters())

    @property
    def core_model(self) -> nn.Module:
        """Gets the core PyTorch model."""
        return self._core_model

    @property
    def optimizer(self) -> torch.optim.Adam:
        """Gets the optimizer."""
        return self._optimizer

    def collate_batch(self, input_tuples: list[tuple[str, int, str, Optional[float]]]
                      ) -> tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """Collate antibody and/or antigen sequences and escape scores for input to the model.

        :param input_tuples: A list of tuples of antibody, site, mutation, and escape score (if available).
        :return: A tuple with a LongTensor containing the antibody and/or antigen sequences
                 and a FloatTensor of escape scores (if available).
        """
        # Unpack list of tuples
        antibodies, sites, mutants, escapes = zip(*input_tuples)

        # Get site indices
        site_indices = torch.LongTensor(sites) - RBD_START_SITE

        # Set up wildtype sequence indices
        wildtype_indices = torch.LongTensor(RBD_AA_INDICES)

        # Get mutant sequence indices
        all_mutant_indices = []
        for site_index, mutant in zip(site_indices, mutants):
            mutant_indices = wildtype_indices.clone()
            mutant_indices[site_index] = AA_TO_INDEX[mutant]
            all_mutant_indices.append(mutant_indices)

        # Stack mutant sequence indices as (sequence_length, num_sequences)
        batch_data = torch.stack(all_mutant_indices).transpose(1, 0)

        # Add the index of the mutation as the first dimension
        batch_data = torch.cat((site_indices.unsqueeze(dim=0), batch_data))

        # Convert escape scores to FloatTensor
        if escapes[0] is not None:
            escapes = torch.FloatTensor(escapes)

            # Optionally binarize escape scores
            if self.binarize:
                escapes = (escapes > 0).float()

        return batch_data, escapes
