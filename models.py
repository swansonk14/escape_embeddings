"""Model classes for predicting escape scores."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


from constants import (
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_LAYER_DIMS,
    DEFAULT_NUM_EPOCHS,
    EMBEDDING_GRANULARITY_OPTIONS,
    HEAVY_CHAIN,
    LIGHT_CHAIN,
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

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return self.__class__.__name__


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

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return self.__class__.__name__


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

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return self.__class__.__name__


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


# TODO: better name
class EmbeddingCoreModel(nn.Module):
    """The core neural model that predicts escape scores by using antibody/antigen embeddings."""

    def __init__(self,
                 binarize: bool,
                 antibody_attention: bool,
                 input_dim: int,
                 hidden_layer_dims: tuple[int, ...]) -> None:
        """Initialize the model.

        TODO: params docstring
        """
        super(EmbeddingCoreModel, self).__init__()

        self.binarize = binarize
        self.antibody_attention = antibody_attention
        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims

        self.output_dim = 1

        # Create attention model
        if self.antibody_attention:
            self.attention = Attention()  # TODO
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
            x = self.attention(x)

        # Apply MLP
        x = self.mlp(x)

        # Apply sigmoid if appropriate
        if not self.training and self.binarize:
            x = self.sigmoid(x)

        # Squeeze output dimension
        x = x.squeeze(dim=1)

        return x


class EmbeddingModel(EscapeModel):
    """A model that predicts escape scores by using antibody/antigen embeddings."""

    def __init__(self,
                 task_type: TASK_TYPE_OPTIONS,
                 embedding_granularity: EMBEDDING_GRANULARITY_OPTIONS,
                 antigen_embeddings: dict[str, torch.FloatTensor],
                 antigen_embedding_type: ANTIGEN_EMBEDDING_TYPE_OPTIONS,
                 antibody_embeddings: Optional[dict[str, torch.FloatTensor]] = None,
                 antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
                 hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
                 num_epochs: int = DEFAULT_NUM_EPOCHS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 model_seed: int = 0) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        TODO: document remaining parameters
        """
        super(EmbeddingModel, self).__init__(task_type=task_type)

        assert (antibody_embeddings is None) == (antibody_embedding_type is None)

        self.embedding_granularity = embedding_granularity
        self.antigen_embeddings = antigen_embeddings
        self.antigen_embedding_type = antigen_embedding_type
        self.antibody_embeddings = antibody_embeddings
        self.antibody_embedding_type = antibody_embedding_type
        self.hidden_layer_dims = hidden_layer_dims
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_seed = model_seed

        # Get embedding dimensionalities
        self.antigen_embedding_dim = next(iter(self.antigen_embeddings.values())).shape[-1]

        if self.antibody_embeddings is not None:
            self.antibody_embedding_dim = next(iter(self.antibody_embeddings.values())).shape[-1]

            # TODO: this might not be necessary and/or could use alternate dimensionality for attention
            if self.antibody_embedding_dim == 'attention':
                assert self.antigen_embedding_dim == self.antibody_embedding_dim
        else:
            self.antibody_embedding_dim = None

        # Optionally compute average sequence embeddings
        if self.embedding_granularity == 'sequence':
            self.antigen_embeddings = {
                name: embedding.mean(dim=0)
                for name, embedding in self.antigen_embeddings.items()
            }

            if self.antibody_embeddings is not None:
                self.antibody_embeddings = {
                    name: embedding.mean(dim=0)
                    for name, embedding in self.antibody_embeddings.items()
                }

        # Get wildtype embedding
        self.wildtype_embedding = self.antigen_embeddings['wildtype']

        # Set up input and output dims
        if self.antibody_embeddings is not None and self.antibody_embedding_type == 'concatenation':
            self.input_dim = self.antigen_embedding_dim + self.antibody_embedding_dim
            # TODO: handle attention
        else:
            self.input_dim = self.antigen_embedding_dim

        # Ensure PyTorch reproducibility
        torch.manual_seed(self.model_seed)
        torch.use_deterministic_algorithms(True)

        # Create core model
        self.embedding_core_model = EmbeddingCoreModel(
            binarize=self.binarize,
            antibody_attention=self.antibody_embedding_type == 'attention',
            input_dim=self.input_dim,
            hidden_layer_dims=self.hidden_layer_dims
        )

        # Create loss function
        if self.task_type == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f'Task type "{self.task_type}" is not supported.')

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.embedding_core_model.parameters())

    def collate_embeddings_and_escape(self,
                                      input_tuples: list[tuple[str, int, str, Optional[float]]]
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
                site_index if self.embedding_granularity == 'residue' else slice(None)]
            for site, mutant, site_index in zip(sites, mutants, site_indices)
        ])

        # Optionally convert mutant embeddings to difference embeddings
        if self.antigen_embedding_type == 'mutant':
            batch_antigen_embeddings = batch_antigen_mutant_embeddings
        elif self.antigen_embedding_type == 'difference':
            batch_antigen_embeddings = (
                    batch_antigen_mutant_embeddings
                    - self.wildtype_embedding[site_indices if self.embedding_granularity == 'residue' else slice(None)]
            )
        else:
            raise ValueError(f'Antigen embedding type "{self.antigen_embedding_type}" is not supported.')

        # Optionally add antibody embeddings to antigen embeddings
        if self.antibody_embeddings is not None:
            batch_antibody_embeddings = torch.stack([
                torch.cat((self.antibody_embeddings[f'{antibody}_{HEAVY_CHAIN}'],
                           self.antibody_embeddings[f'{antibody}_{LIGHT_CHAIN}']))
                for antibody in antibodies
            ])

            if self.antibody_embedding_type == 'concatenation':
                batch_embeddings = torch.cat((batch_antigen_embeddings, batch_antibody_embeddings), dim=1)
            elif self.antibody_embedding_type == 'attention':
                raise NotImplementedError  # TODO: implement this
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

    def fit(self,
            antibodies: list[str],
            sites: list[int],
            wildtypes: list[str],
            mutants: list[str],
            escapes: list[float]) -> 'EmbeddingModel':
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
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # TODO: enable parallel workers
            collate_fn=self.collate_embeddings_and_escape
        )

        # Train
        for _ in trange(self.num_epochs, desc='Epochs', leave=False):
            for batch_embeddings, batch_escape in tqdm(data_loader, total=len(data_loader), desc='Batches', leave=False):
                self.embedding_core_model.zero_grad()

                preds = self.embedding_core_model(batch_embeddings)

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
            num_workers=0,  # TODO: enable parallel workers
            collate_fn=self.collate_embeddings_and_escape
        )

        # Predict
        all_preds = []

        with torch.no_grad():
            for batch_embeddings, _ in tqdm(data_loader, total=len(data_loader), desc='Batches', leave=False):
                preds = self.embedding_core_model(batch_embeddings)
                all_preds.append(preds.numpy())

        all_preds = np.concatenate(all_preds)

        return all_preds

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return str(self.embedding_core_model)
