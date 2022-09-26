"""Model classes for predicting escape scores."""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
import torch

from constants import (
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_HIDDEN_LAYER_SIZES,
    EMBEDDING_GRANULARITY_OPTIONS,
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
                mutants: Iterable[str],) -> np.ndarray:
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

        self.wildtype_to_mutant_to_escape_list = defaultdict(lambda: defaultdict(list))
        self.wildtype_to_mutant_to_average_escape = defaultdict(lambda: defaultdict(float))

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
            self.wildtype_to_mutant_to_escape_list[wildtype][mutant].append(int(escape > 0) if self.binarize else escape)

        for wildtype, mutant_to_escape_list in self.wildtype_to_mutant_to_escape_list.items():
            for mutant, escape_list in mutant_to_escape_list.items():
                self.wildtype_to_mutant_to_average_escape[wildtype][mutant] = sum(escape_list) / len(escape_list)

        return self

    def predict(self,
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str],) -> np.ndarray:
        """Predict the average escape score of each wildtype-mutation amino acid substitution.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array containing average escape scores for each wildtype-mutation amino acid substitution.
        """
        return np.array([
            self.wildtype_to_mutant_to_average_escape[wildtype][mutant]
            for wildtype, mutant in zip(wildtypes, mutants)
        ])


class SiteModel(EscapeModel):
    """A model that predicts the average escape score at each antigen site."""

    def __init__(self, task_type: TASK_TYPE_OPTIONS) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        """
        super(SiteModel, self).__init__(task_type=task_type)

        self.site_to_escape_list = defaultdict(list)
        self.site_to_average_escape = defaultdict(float)

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
            self.site_to_escape_list[site].append(int(escape > 0) if self.binarize else escape)

        for site, escape_list in self.site_to_escape_list.items():
            self.site_to_average_escape[site] = sum(escape_list) / len(escape_list)

        return self

    def predict(self,
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str],) -> np.ndarray:
        """Predict the average escape score at each antigen site.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array containing average escape scores for each antigen site.
        """
        return np.array([self.site_to_average_escape[site] for site in sites])


class LikelihoodModel(EscapeModel):
    """A model that predicts escape scores by using antigen mutant vs wildtype likelihood."""

    def __init__(self, task_type: TASK_TYPE_OPTIONS, antigen_likelihoods: dict[str, float]) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        :param antigen_likelihoods: A dictionary mapping from antigen name to mutant vs wildtype likelihood.
        """
        super(LikelihoodModel, self).__init__(task_type=task_type)

        self.antigen_likelihoods = antigen_likelihoods

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
        return self

    @abstractmethod
    def predict(self,
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str],) -> np.ndarray:
        """Makes escape predictions on the test data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array of predicted escape scores.
        """
        return np.array([self.antigen_likelihoods[f'{site}_{mutant}'] for site, mutant in zip(sites, mutants)])


class EmbeddingModel(EscapeModel):
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

        self.embedding_granularity = embedding_granularity
        self.antigen_embeddings = antigen_embeddings
        self.antigen_embedding_type = antigen_embedding_type
        self.antibody_embeddings = antibody_embeddings
        self.antibody_embedding_type = antibody_embedding_type
        self.hidden_layer_sizes = hidden_layer_sizes

        # TODO: create model and handle embedding granularity

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
        raise NotImplementedError  # TODO

    @abstractmethod
    def predict(self,
                antibodies: Iterable[str],
                sites: Iterable[int],
                wildtypes: Iterable[str],
                mutants: Iterable[str],) -> np.ndarray:
        """Makes escape predictions on the test data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array of predicted escape scores.
        """
        raise NotImplementedError  # TODO
