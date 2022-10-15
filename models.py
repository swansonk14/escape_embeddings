"""Model classes for predicting escape scores."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


from constants import (
    AA_TO_INDEX,
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_ATTENTION_NUM_HEADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_HIDDEN_LAYER_DIMS,
    EMBEDDING_GRANULARITY_OPTIONS,
    HEAVY_CHAIN,
    LIGHT_CHAIN,
    RBD_AA_INDICES,
    RBD_START_SITE,
    TASK_TYPE_OPTIONS
)


class EscapeModel(ABC):
    """Abstract class for an escape prediction model."""

    def __init__(self, task_type: TASK_TYPE_OPTIONS) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        """
        self.task_type = task_type

        if self.task_type not in {'classification', 'regression'}:
            raise ValueError(f'Task type "{task_type}" is not supported.')

        self.binarize = self.task_type == 'classification'

    @abstractmethod
    def fit(self,
            antibodies: list[str],
            sites: list[int],
            wildtypes: list[str],
            mutants: list[str],
            escapes: list[float]) -> 'EscapeModel':
        """Fits the model on the training escape data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        :return: The fitted model.
        """
        pass

    @abstractmethod
    def predict(self,
                antibodies: list[str],
                sites: list[int],
                wildtypes: list[str],
                mutants: list[str]) -> np.ndarray:
        """Makes escape predictions on the test data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array of predicted escape scores.
        """
        pass

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return self.__class__.__name__


class MutationModel(EscapeModel):
    """A model that predicts the average escape score of each wildtype-mutant amino acid substitution."""

    def __init__(self, task_type: TASK_TYPE_OPTIONS) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        """
        super(MutationModel, self).__init__(task_type=task_type)

        self.wildtype_to_mutant_to_escape_list = {}
        self.wildtype_to_mutant_to_average_escape = None

    def fit(self,
            antibodies: list[str],
            sites: list[int],
            wildtypes: list[str],
            mutants: list[str],
            escapes: list[float]) -> 'MutationModel':
        """Fit the model by computing the average escape score of each wildtype-mutation amino acid substitution.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        :return: The fitted model.
        """
        for wildtype, mutant, escape in zip(wildtypes, mutants, escapes):
            escape = int(escape > 0) if self.binarize else escape
            self.wildtype_to_mutant_to_escape_list.setdefault(wildtype, {}).setdefault(mutant, []).append(escape)

        self.wildtype_to_mutant_to_average_escape = {
            wildtype: {
                mutant: sum(escape_list) / len(escape_list)
                for mutant, escape_list in mutant_to_escape_list.items()
            }
            for wildtype, mutant_to_escape_list in self.wildtype_to_mutant_to_escape_list.items()
        }

        return self

    def predict(self,
                antibodies: list[str],
                sites: list[int],
                wildtypes: list[str],
                mutants: list[str]) -> np.ndarray:
        """Predict the average escape score of each wildtype-mutation amino acid substitution.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array containing average escape scores for each wildtype-mutation amino acid substitution.
        """
        return np.array([
            self.wildtype_to_mutant_to_average_escape.get(wildtype, {}).get(mutant, 0.0)
            for wildtype, mutant in zip(wildtypes, mutants)
        ])


class SiteModel(EscapeModel):
    """A model that predicts the average escape score at each antigen site."""

    def __init__(self, task_type: TASK_TYPE_OPTIONS) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        """
        super(SiteModel, self).__init__(task_type=task_type)

        self.site_to_escape_list = {}
        self.site_to_average_escape = None

    def fit(self,
            antibodies: list[str],
            sites: list[int],
            wildtypes: list[str],
            mutants: list[str],
            escapes: list[float]) -> 'SiteModel':
        """Fit the model by computing the average escape score at each antigen site.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        :return: The fitted model.
        """
        for site, escape in zip(sites, escapes):
            escape = int(escape > 0) if self.binarize else escape
            self.site_to_escape_list.setdefault(site, []).append(escape)

        self.site_to_average_escape = {
            site: sum(escape_list) / len(escape_list)
            for site, escape_list in self.site_to_escape_list.items()
        }

        return self

    def predict(self,
                antibodies: list[str],
                sites: list[int],
                wildtypes: list[str],
                mutants: list[str]) -> np.ndarray:
        """Predict the average escape score at each antigen site.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array containing average escape scores for each antigen site.
        """
        return np.array([self.site_to_average_escape.get(site, 0.0) for site in sites])


class LikelihoodModel(EscapeModel):
    """A model that predicts escape scores by using antigen mutant vs wildtype likelihood."""

    def __init__(self, task_type: TASK_TYPE_OPTIONS, antigen_likelihoods: dict[str, float]) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        :param antigen_likelihoods: A dictionary mapping from antigen name to mutant vs wildtype likelihood.
        """
        super(LikelihoodModel, self).__init__(task_type=task_type)

        self.antigen_likelihoods = antigen_likelihoods

    def fit(self,
            antibodies: list[str],
            sites: list[int],
            wildtypes: list[str],
            mutants: list[str],
            escapes: list[float]) -> 'LikelihoodModel':
        """Fits the model on the training escape data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        :return: The fitted model.
        """
        return self

    def predict(self,
                antibodies: list[str],
                sites: list[int],
                wildtypes: list[str],
                mutants: list[str]) -> np.ndarray:
        """Makes escape predictions on the test data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array of predicted escape scores.
        """
        return np.array([self.antigen_likelihoods[f'{site}_{mutant}'] for site, mutant in zip(sites, mutants)])


class MutationDataset(Dataset):
    """A data set for mutations containing antibodies, sites, and mutations."""

    def __init__(self,
                 antibodies: list[str],
                 sites: list[int],
                 mutants: list[str],
                 escapes: Optional[list[float]] = None) -> None:
        """Initialize the dataset.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        """
        escapes = escapes if escapes is not None else [None] * len(antibodies)

        assert len(antibodies) == len(sites) == len(mutants) == len(escapes)

        self.antibodies = antibodies
        self.sites = sites
        self.mutants = mutants
        self.escapes = escapes

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.antibodies)

    def __getitem__(self, item: int | slice) -> tuple[str, int, str, Optional[float]] \
                                                | list[tuple[str, int, str, Optional[float]]]:
        """Get an item from the dataset

        :param item: An int or slice to index into the dataset.
        :return: An item or list of items from the dataset corresponding to the provided index or slice.
                 Each item is a tuple of an antibody, a site, a mutation, and an escape score (if available).
        """
        if isinstance(item, int):
            return self.antibodies[item], self.sites[item], self.mutants[item], self.escapes[item]

        if isinstance(item, slice):
            return list(zip(*[self.antibodies[item], self.sites[item], self.mutants[item], self.escapes[item]]))

        raise NotImplementedError(f'__getitem__ with item type "{type(item)}" is not supported.')


class PyTorchEscapeModel(EscapeModel):
    """An abstract PyTorch model that predicts escape."""

    def __init__(self,
                 task_type: TASK_TYPE_OPTIONS,
                 num_epochs: int,
                 batch_size: int,
                 device: str) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        """
        super(PyTorchEscapeModel, self).__init__(task_type=task_type)

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        # Ensure PyTorch reproducibility
        torch.manual_seed(0)

        # Create loss function
        if self.task_type == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f'Task type "{self.task_type}" is not supported.')

    @property
    @abstractmethod
    def core_model(self) -> nn.Module:
        """Gets the core PyTorch model."""
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Adam:
        """Gets the optimizer."""
        pass

    @abstractmethod
    def collate_batch(self, input_tuples: list[tuple[str, int, str, Optional[float]]]
                      ) -> tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        """Collate antibody and/or antigen sequences/embeddings and escape scores for input to the model.

        :param input_tuples: A list of tuples of antibody, site, mutation, and escape score (if available).
        :return: A tuple with a Tensor containing the antibody and/or antigen sequences/embeddings
                 and a FloatTensor of escape scores (if available).
        """
        pass

    def fit(self,
            antibodies: list[str],
            sites: list[int],
            wildtypes: list[str],
            mutants: list[str],
            escapes: list[float]) -> 'PyTorchEscapeModel':
        """Fits the model on the training escape data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        :return: The fitted model.
        """
        # Create mutation dataset
        dataset = MutationDataset(
            antibodies=antibodies,
            sites=sites,
            mutants=mutants,
            escapes=escapes
        )

        # Create data loader
        generator = torch.Generator()
        generator.manual_seed(0)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
            generator=generator
        )

        # Train
        self.core_model.train()
        for _ in trange(self.num_epochs, desc='Epochs', leave=False):
            for batch_data, batch_escape in tqdm(data_loader, total=len(data_loader), desc='Batches', leave=False):
                self.core_model.zero_grad()

                batch_data = batch_data.to(self.device)
                batch_escape = batch_escape.to(self.device)

                preds = self.core_model(batch_data)

                loss = self.loss_fn(preds, batch_escape)

                loss.backward()
                self.optimizer.step()

        return self

    def predict(self,
                antibodies: list[str],
                sites: list[int],
                wildtypes: list[str],
                mutants: list[str]) -> np.ndarray:
        """Makes escape predictions on the test data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array of predicted escape scores.
        """
        # Create mutation dataset
        dataset = MutationDataset(
            antibodies=antibodies,
            sites=sites,
            mutants=mutants
        )

        # Create data loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch
        )

        # Predict
        all_preds = []

        self.core_model.eval()
        with torch.no_grad():
            for batch_data, _ in tqdm(data_loader, total=len(data_loader), desc='Batches', leave=False):
                batch_data = batch_data.to(self.device)
                preds = self.core_model(batch_data)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        return all_preds

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return str(self.core_model)


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
        """TODO: docstring"""
        # Apply layers
        for i, layer in enumerate(self.layers):
            # TODO: dropout or batch norm?
            x = layer(x)

            if i != len(self.layers) - 1:
                x = self.activation(x)

        return x


class Attention(nn.Module):
    """An attention model."""

    def __init__(self) -> None:
        """Initialize the model."""
        super(Attention, self).__init__()

    def forward(self) -> torch.FloatTensor:
        """TODO: docstring"""
        raise NotImplementedError


class EmbeddingCoreModel(nn.Module):
    """The core neural model that predicts escape scores by using antibody/antigen embeddings."""

    def __init__(self,
                 binarize: bool,
                 antibody_attention: bool,
                 input_dim: int,
                 hidden_layer_dims: tuple[int, ...],
                 attention_num_heads: int) -> None:
        """Initialize the model.

        TODO: params docstring
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
            self.attention = nn.MultiheadAttention(
                embed_dim=self.input_dim // 3,
                num_heads=self.attention_num_heads
            )
        else:
            self.attention = None

        # Create MLP model
        self.mlp = MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layer_dims=self.hidden_layer_dims
        )

        # Create sigmoid function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """TODO: docstring"""
        # Apply attention
        if self.antibody_attention:
            batch_size = x.shape[1]
            x, x_weights = self.attention(x, x, x)  # (3, batch_size, embedding_size), (batch_size, 3, 3)
            x = torch.transpose(x, 0, 1).reshape(batch_size, -1)  # (batch_size, 3 * embedding_size)

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
                 hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
                 attention_num_heads: int = DEFAULT_ATTENTION_NUM_HEADS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str = DEFAULT_DEVICE) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        TODO: document remaining parameters
        """
        super(EmbeddingModel, self).__init__(
            task_type=task_type,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        if antibody_embedding_granularity is not None and antibody_embedding_granularity != 'sequence':
            raise NotImplementedError(f'Antibody embedding granularity "{antibody_embedding_granularity}" '
                                      f'has not been implemented yet.')

        assert (antibody_embeddings is None) == (antibody_embedding_type is None)

        self.antigen_embeddings = antigen_embeddings
        self.antigen_embedding_granularity = antigen_embedding_granularity
        self.antigen_embedding_type = antigen_embedding_type
        self.antibody_embeddings = antibody_embeddings
        self.antibody_embedding_granularity = antibody_embedding_granularity
        self.antibody_embedding_type = antibody_embedding_type
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
        if self.antigen_embedding_granularity == 'sequence':
            self.antigen_embeddings = {
                name: embedding.mean(dim=0)
                for name, embedding in self.antigen_embeddings.items()
            }

        if self.antibody_embeddings is not None and self.antibody_embedding_granularity == 'sequence':
            self.antibody_embeddings = {
                name: embedding.mean(dim=0)
                for name, embedding in self.antibody_embeddings.items()
            }

        # Get wildtype embedding
        self.wildtype_embedding = self.antigen_embeddings['wildtype']

        # Set up input and output dims
        self.input_dim = (1 + (self.antigen_embedding_type == 'mutant_difference')) * self.antigen_embedding_dim

        if self.antibody_embeddings is not None:
            self.input_dim += 2 * self.antibody_embedding_dim  # heavy and light chain

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
                      ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
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
        batch_antigen_mutant_embeddings = torch.stack([
            self.antigen_embeddings[f'{site}_{mutant}'][
                site_index if self.antigen_embedding_granularity == 'residue' else slice(None)
            ]
            for site, mutant, site_index in zip(sites, mutants, site_indices)
        ])

        # Optionally convert mutant embeddings to difference embeddings
        if self.antigen_embedding_type == 'mutant':
            batch_antigen_embeddings = batch_antigen_mutant_embeddings
        elif self.antigen_embedding_type in {'difference', 'mutant_difference'}:
            batch_difference_embeddings = (
                    batch_antigen_mutant_embeddings
                    - self.wildtype_embedding[site_indices if self.antigen_embedding_granularity == 'residue' else slice(None)]
            )

            if self.antigen_embedding_type == 'difference':
                batch_antigen_embeddings = batch_difference_embeddings
            elif self.antigen_embedding_type == 'mutant_difference':
                batch_antigen_embeddings = torch.cat(
                    (batch_antigen_mutant_embeddings, batch_difference_embeddings), dim=1
                )
            else:
                raise ValueError(f'Antigen embedding type "{self.antigen_embedding_type}" is not supported.')
        else:
            raise ValueError(f'Antigen embedding type "{self.antigen_embedding_type}" is not supported.')

        # Optionally add antibody embeddings to antigen embeddings
        if self.antibody_embeddings is not None:
            batch_antibody_heavy_embeddings = torch.stack([
                self.antibody_embeddings[f'{antibody}_{HEAVY_CHAIN}']
                for antibody in antibodies
            ])
            batch_antibody_light_embeddings = torch.stack([
                self.antibody_embeddings[f'{antibody}_{LIGHT_CHAIN}']
                for antibody in antibodies
            ])

            if self.antibody_embedding_type == 'concatenation':
                batch_embeddings = torch.cat(
                    (batch_antigen_embeddings, batch_antibody_heavy_embeddings, batch_antibody_light_embeddings), dim=1
                )
            elif self.antibody_embedding_type == 'attention':
                batch_embeddings = torch.stack(
                    (batch_antigen_embeddings, batch_antibody_heavy_embeddings, batch_antibody_light_embeddings)
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


class RNNCoreModel(nn.Module):
    """The core recurrent neural network that predicts escape scores based on one-hot amino acid features."""

    def __init__(self,
                 hidden_dim: int,
                 hidden_layer_dims: tuple[int, ...]) -> None:
        """Initialize the model.

        TODO: params docstring
        """
        super(RNNCoreModel, self).__init__()

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
        """TODO: docstring"""
        # Extract mutation sites
        site_indices, x = x[0], x[1:]

        # Get batch size
        batch_size = x.shape[1]

        # Embed amino acids
        x = self.embeddings(x)

        # Run RNN
        output, _ = self.rnn(x)  # (sequence_length, batch_size, 2 * hidden_dim)
        x = output[site_indices, torch.arange(batch_size)]

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
                 num_epochs: int,
                 hidden_dim: int,
                 hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str = DEFAULT_DEVICE) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        TODO: document remaining parameters
        """
        super(RNNModel, self).__init__(
            task_type=task_type,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        self.hidden_dim = hidden_dim
        self.hidden_layer_dims = hidden_layer_dims

        # Create model
        self._core_model = RNNCoreModel(
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
