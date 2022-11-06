"""Contains statistical baseline models based on mutation and site for predicting escape."""

import numpy as np

from constants import TASK_TYPE_OPTIONS
from models_escape import EscapeModel


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
