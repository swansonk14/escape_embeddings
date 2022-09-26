"""Model classes for predicting escape scores."""
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from constants import (
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_HIDDEN_LAYER_SIZES,
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
        super(EscapeModel, self).__init__()

        self.task_type = task_type

        if self.task_type not in {'classification', 'regression'}:
            raise ValueError(f'Task type "{task_type}" is not supported.')

        self.binarize = self.task_type == 'classification'

    @abstractmethod
    def fit(self,
            antibodies: Iterable[str],
            sites: Iterable[int],
            wildtypes: Iterable[str],
            mutants: Iterable[str],
            escapes: Iterable[float]) -> 'EscapeModel':
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
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str]) -> np.ndarray:
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
            antibodies: Iterable[str],
            sites: Iterable[int],
            wildtypes: Iterable[str],
            mutants: Iterable[str],
            escapes: Iterable[float]) -> 'MutationModel':
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
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str]) -> np.ndarray:
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
            antibodies: Iterable[str],
            sites: Iterable[int],
            wildtypes: Iterable[str],
            mutants: Iterable[str],
            escapes: Iterable[float]) -> 'SiteModel':
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
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str]) -> np.ndarray:
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
            antibodies: Iterable[str],
            sites: Iterable[int],
            wildtypes: Iterable[str],
            mutants: Iterable[str],
            escapes: Iterable[float]) -> 'EscapeModel':
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
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str]) -> np.ndarray:
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


class EmbeddingModel(EscapeModel, nn.Module):
    """A model that predicts escape scores by using antibody/antigen embeddings."""

    def __init__(self,
                 task_type: TASK_TYPE_OPTIONS,
                 embedding_granularity: EMBEDDING_GRANULARITY_OPTIONS,
                 antigen_embeddings: dict[str, torch.FloatTensor],
                 antigen_embedding_type: ANTIGEN_EMBEDDING_TYPE_OPTIONS,
                 antibody_embeddings: Optional[dict[str, torch.FloatTensor]] = None,
                 antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
                 hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES) -> None:
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
        self.hidden_layer_sizes = hidden_layer_sizes

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

        # Set up layer dimensionalities
        if self.antibody_embeddings is not None and self.antibody_embedding_type == 'concatenation':
            self.input_dim = self.antigen_embedding_dim + self.antibody_embedding_dim
        else:
            self.input_dim = self.antigen_embedding_dim

        self.output_dim = 1
        self.layer_dims = [self.input_dim] + list(hidden_layer_sizes) + [self.output_dim]

        # Create layers
        self.layers = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            for i in range(len(self.layer_dims) - 1)
        ])

        # Create activation function
        self.activation = nn.ReLU()

        # Create output function
        if self.binarize:
            self.output_function = nn.Sigmoid()
        else:
            self.output_function = None

        # TODO: create model and handle embedding granularity

    def fit(self,
            antibodies: Iterable[str],
            sites: Iterable[int],
            wildtypes: Iterable[str],
            mutants: Iterable[str],
            escapes: Iterable[float]) -> 'EscapeModel':
        """Fits the model on the training escape data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        :return: The fitted model.
        """
        self.forward(
            antibodies=antibodies,
            sites=sites,
            wildtypes=wildtypes,
            mutants=mutants
        )
        raise NotImplementedError  # TODO

    def predict(self,
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str]) -> np.ndarray:
        """Makes escape predictions on the test data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array of predicted escape scores.
        """
        raise NotImplementedError  # TODO

    def forward(self,
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str]) -> torch.FloatTensor:
        """TODO: docstring"""
        # Get site indices
        site_indices = torch.LongTensor(sites) - RBD_START_SITE

        # Get antigen mutant embeddings for the batch
        batch_antigen_mutant_embeddings = torch.stack([
            self.antigen_embeddings[f'{site}_{mutant}'][site_index if self.embedding_granularity == 'residue' else slice(None)]
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
                breakpoint()  # TODO
                pass
            else:
                raise ValueError(f'Antibody embedding type "{self.antibody_embedding_type} is not supported.')
        else:
            batch_embeddings = batch_antigen_embeddings

        breakpoint()
