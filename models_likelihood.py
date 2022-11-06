"""Contains an embedding-based, zero-shot likelihood model."""
import numpy as np

from constants import TASK_TYPE_OPTIONS
from models_escape import EscapeModel


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
