"""Defines an abstract escape model class."""
from abc import ABC, abstractmethod

import numpy as np

from constants import TASK_TYPE_OPTIONS


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
