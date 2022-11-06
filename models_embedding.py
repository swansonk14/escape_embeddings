"""Contains embedding model classes for predicting escape."""
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from constants import (
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_ATTENTION_NUM_HEADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_HIDDEN_LAYER_DIMS,
    EMBEDDING_GRANULARITY_OPTIONS,
    HEAVY_CHAIN,
    LIGHT_CHAIN,
    RBD_START_SITE,
    TASK_TYPE_OPTIONS
)
from models_mlp import MLP
from models_pytorch import PyTorchEscapeModel


class EmbeddingCoreModel(nn.Module):
    """The core neural model that predicts escape scores by using antibody/antigen embeddings."""

    def __init__(self,
                 binarize: bool,
                 antibody_attention: bool,
                 input_dim: int,
                 hidden_layer_dims: tuple[int, ...],
                 attention_num_heads: int) -> None:
        """Initialize the model.

        :param binarize: Whether the escape scores are binarized (for classification), thus requiring a sigmoid.
        :param antibody_attention: Whether to apply antibody attention.
        :param input_dim: The dimensionality of the input to the model.
        :param hidden_layer_dims: The dimensionalities of the hidden layers.
        :param attention_num_heads: The number of attention heads to use.
        """
        super(EmbeddingCoreModel, self).__init__()

        self.binarize = binarize
        self.antibody_attention = antibody_attention
        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.attention_num_heads = attention_num_heads

        self.output_dim = 1

        # Create attention model
        if self.antibody_attention:
            self.embed_dim = self.input_dim // 3
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.attention_num_heads,
                batch_first=True
            )
            mlp_input_dim = self.embed_dim * 2
        else:
            self.attention = self.embed_dim = None
            mlp_input_dim = self.input_dim

        # Create MLP model
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            output_dim=self.output_dim,
            hidden_layer_dims=self.hidden_layer_dims
        )

        # Create sigmoid function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.FloatTensor | tuple[torch.FloatTensor, ...]) -> torch.FloatTensor:
        """Runs the model on the data.

        :param x: A FloatTensor containing an embedding of the antibody and/or antigen
                  or a tuple of FloatTensors containing the antibody chain and antigen embeddings if using attention.
        :return: A FloatTensor containing the model's predicted escape score.
        """
        # Apply attention
        if self.antibody_attention:
            antigen_embeddings, antibody_heavy_embeddings, antibody_light_embeddings = x
            antigen_heavy_embeddings, _ = self.attention(
                antigen_embeddings, antibody_heavy_embeddings, antibody_heavy_embeddings
            )  # (batch_size, antigen_len, embedding_size)
            antigen_light_embeddings, _ = self.attention(
                antigen_embeddings, antibody_light_embeddings, antibody_light_embeddings
            )  # (batch_size, antigen_len, embedding_size)
            x = torch.cat(
                (antigen_heavy_embeddings.mean(dim=1), antigen_light_embeddings.mean(dim=1)),
                dim=-1
            )  # (batch_size, 2 * embedding_size)

        # Apply MLP
        x = self.mlp(x)

        # Apply sigmoid if appropriate
        if not self.training and self.binarize:
            x = self.sigmoid(x)

        # Squeeze output dimension
        x = x.squeeze(dim=1)

        return x


class EmbeddingModel(PyTorchEscapeModel):
    """A model that predicts escape scores by using antibody/antigen embeddings."""

    def __init__(self,
                 task_type: TASK_TYPE_OPTIONS,
                 num_epochs: int,
                 antigen_embeddings: dict[str, torch.FloatTensor],
                 antigen_embedding_granularity: EMBEDDING_GRANULARITY_OPTIONS,
                 antigen_embedding_type: ANTIGEN_EMBEDDING_TYPE_OPTIONS,
                 antibody_embeddings: Optional[dict[str, torch.FloatTensor]] = None,
                 antibody_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
                 antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
                 unique_antibodies: Optional[list[str]] = None,
                 hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
                 attention_num_heads: int = DEFAULT_ATTENTION_NUM_HEADS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str = DEFAULT_DEVICE) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        :param num_epochs: The number of epochs to train for.
        :param antigen_embeddings: A dictionary mapping from antigen name to ESM2 embedding.
        :param antigen_embedding_granularity: The granularity of the antigen embeddings, either a sequence average or per-residue embeddings.
        :param antigen_embedding_type: The type of antigen embedding.
        :param antibody_embeddings: A dictionary mapping from antibody name_chain to ESM2 embedding.
        :param antibody_embedding_granularity: The granularity of the antibody embeddings, either a sequence average or per-residue embeddings.
        :param antibody_embedding_type: Method of including the antibody embeddings with antigen embeddings.
        :param unique_antibodies: A list of unique antibodies.
        :param hidden_layer_dims: The sizes of the hidden layers of the MLP model that will be trained.
        :param attention_num_heads: The number of attention heads for the attention antibody embedding type.
        :param num_epochs: The number of epochs for the embedding model. If None, num_epochs is set based on model_granularity.
        :param batch_size: The batch size for the embedding model.
        :param device: The device to use (e.g., "cpu" or "cuda") for the RNN and embedding models.
        """
        super(EmbeddingModel, self).__init__(
            task_type=task_type,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        assert (antibody_embeddings is None) == (antibody_embedding_type in {None, 'one_hot'})

        self.antigen_embeddings = antigen_embeddings
        self.antigen_embedding_granularity = antigen_embedding_granularity
        self.antigen_embedding_type = antigen_embedding_type
        self.antibody_embeddings = antibody_embeddings
        self.antibody_embedding_granularity = antibody_embedding_granularity
        self.antibody_embedding_type = antibody_embedding_type
        self.unique_antibodies = unique_antibodies
        self.hidden_layer_dims = hidden_layer_dims
        self.attention_num_heads = attention_num_heads

        # Get embedding dimensionalities
        self.antigen_embedding_dim = next(iter(self.antigen_embeddings.values())).shape[-1]

        if self.antibody_embeddings is not None:
            self.antibody_embedding_dim = next(iter(self.antibody_embeddings.values())).shape[-1]

            if self.antibody_embedding_dim == 'attention':
                assert self.antigen_embedding_dim == self.antibody_embedding_dim
        else:
            self.antibody_embedding_dim = None

        # Optionally compute average sequence embeddings
        if self.antigen_embedding_granularity == 'sequence' and self.antigen_embedding_type != 'linker':
            self.antigen_embeddings = {
                name: embedding.mean(dim=0)
                for name, embedding in self.antigen_embeddings.items()
            }

        if self.antibody_embeddings is not None and self.antibody_embedding_granularity == 'sequence':
            self.antibody_embeddings = {
                name: embedding.mean(dim=0)
                for name, embedding in self.antibody_embeddings.items()
            }

        # Set up one-hot antibody embeddings
        if self.antibody_embedding_type == 'one_hot':
            eye_matrix = torch.eye(len(self.unique_antibodies))
            self.antibody_embeddings = {
                antibody: eye_matrix[index]
                for index, antibody in enumerate(self.unique_antibodies)
            }
            self.antibody_embedding_dim = len(self.unique_antibodies)

        # Get wildtype embedding
        if self.antigen_embedding_type in {'difference', 'mutant_difference'}:
            self.wildtype_embedding = self.antigen_embeddings['wildtype']
        else:
            self.wildtype_embedding = None

        # Set up input and output dims
        num_antigen_embeddings = (1 + (self.antigen_embedding_type in {'mutant_difference', 'linker'}))
        self.input_dim = num_antigen_embeddings * self.antigen_embedding_dim

        if self.antibody_embeddings is not None:
            # Heavy and light chains if not using one_hot encoding
            self.input_dim += (1 + (self.antibody_embedding_type != 'one_hot')) * self.antibody_embedding_dim

        # Create core model
        self._core_model = EmbeddingCoreModel(
            binarize=self.binarize,
            antibody_attention=self.antibody_embedding_type == 'attention',
            input_dim=self.input_dim,
            hidden_layer_dims=self.hidden_layer_dims,
            attention_num_heads=self.attention_num_heads
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
                      ) -> tuple[torch.FloatTensor | tuple[torch.FloatTensor, ...], Optional[torch.FloatTensor]]:
        """Collate antibody and/or antigen embeddings and escape scores for input to the model.

        :param input_tuples: A list of tuples of antibody, site, mutation, and escape score (if available).
        :return: A tuple with a FloatTensor containing the antibody and/or antigen embeddings
                 and a FloatTensor of escape scores (if available).
        """
        # Unpack list of tuples
        antibodies, sites, mutants, escapes = zip(*input_tuples)

        # Get site indices
        site_indices = torch.LongTensor(sites) - RBD_START_SITE

        # Get antigen mutant embeddings for the batch
        if self.antigen_embedding_type == 'linker':
            batch_antibody_antigen_heavy_embeddings = torch.stack([
                self.antigen_embeddings[f'{antibody}_{HEAVY_CHAIN}_{site}_{mutant}']
                for antibody, site, mutant in zip(antibodies, sites, mutants)
            ])
            batch_antibody_antigen_light_embeddings = torch.stack([
                self.antigen_embeddings[f'{antibody}_{LIGHT_CHAIN}_{site}_{mutant}']
                for antibody, site, mutant in zip(antibodies, sites, mutants)
            ])
            batch_antigen_embeddings = torch.cat(
                (batch_antibody_antigen_heavy_embeddings, batch_antibody_antigen_light_embeddings), dim=1
            )
        else:
            index_into_antigen_embeddings = self.antigen_embedding_granularity == 'residue' \
                                            and self.antibody_embedding_type != 'attention'

            batch_antigen_mutant_embeddings = torch.stack([
                self.antigen_embeddings[f'{site}_{mutant}'][
                    site_index if index_into_antigen_embeddings else slice(None)
                ]
                for site, mutant, site_index in zip(sites, mutants, site_indices)
            ])

            # Optionally convert mutant embeddings to difference embeddings
            if self.antigen_embedding_type in 'mutant':
                batch_antigen_embeddings = batch_antigen_mutant_embeddings
            elif self.antigen_embedding_type in {'difference', 'mutant_difference'}:
                batch_difference_embeddings = (
                        batch_antigen_mutant_embeddings
                        - self.wildtype_embedding[site_indices if index_into_antigen_embeddings else slice(None)]
                )

                if self.antigen_embedding_type == 'difference':
                    batch_antigen_embeddings = batch_difference_embeddings
                elif self.antigen_embedding_type == 'mutant_difference':
                    batch_antigen_embeddings = torch.cat(
                        (batch_antigen_mutant_embeddings, batch_difference_embeddings), dim=-1
                    )
                else:
                    raise ValueError(f'Antigen embedding type "{self.antigen_embedding_type}" is not supported.')
            else:
                raise ValueError(f'Antigen embedding type "{self.antigen_embedding_type}" is not supported.')

        # Optionally add antibody embeddings to antigen embeddings
        if self.antibody_embeddings is not None:
            if self.antibody_embedding_type == 'one_hot':
                batch_antibody_embeddings = torch.stack([
                    self.antibody_embeddings[antibody]
                    for antibody in antibodies
                ])
                batch_embeddings = torch.cat((batch_antigen_embeddings, batch_antibody_embeddings), dim=1)
            else:
                batch_antibody_heavy_embeddings = pad_sequence(
                    [self.antibody_embeddings[f'{antibody}_{HEAVY_CHAIN}'] for antibody in antibodies],
                    batch_first=True
                )
                batch_antibody_light_embeddings = pad_sequence(
                    [self.antibody_embeddings[f'{antibody}_{LIGHT_CHAIN}'] for antibody in antibodies],
                    batch_first=True
                )

                if self.antibody_embedding_type == 'concatenation':
                    batch_embeddings = torch.cat(
                        (batch_antigen_embeddings, batch_antibody_heavy_embeddings, batch_antibody_light_embeddings),
                        dim=1
                    )
                elif self.antibody_embedding_type == 'attention':
                    batch_embeddings = (
                        batch_antigen_embeddings,
                        batch_antibody_heavy_embeddings,
                        batch_antibody_light_embeddings
                    )
                else:
                    raise ValueError(f'Antibody embedding type "{self.antibody_embedding_type} is not supported.')
        else:
            batch_embeddings = batch_antigen_embeddings

        # Convert escape scores to FloatTensor
        if escapes[0] is not None:
            escapes = torch.FloatTensor(escapes)

            # Optionally binarize escape scores
            if self.binarize:
                escapes = (escapes > 0).float()

        return batch_embeddings, escapes
