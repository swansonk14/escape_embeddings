"""Train a model to predict antigen escape using ESM2 embeddings."""
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from tap import Tap
from tqdm import tqdm

from constants import (
    ANTIBODY_COLUMN
)


class SiteModel:
    """A Model that predicts the average escape score at each antigen site."""

    def __init__(self) -> None:
        """Initialize the model."""
        self.site_to_escape_list = defaultdict(list)
        self.site_to_average_escape = defaultdict(float)

    def fit(self, sites: list[int], escapes: list[float], binarize: bool) -> 'SiteModel':
        """Fit the model by computing the average escape score at each antigen site.

        :param sites: A list of mutated sites.
        :param escapes: A list of escape scores corresponding to the sites.
        :param binarize: Whether to binarize the escape into 0 or 1 for non-zero values.
        :return: Returns the fitted model.
        """
        for site, escape in zip(sites, escapes):
            self.site_to_escape_list[site].append(int(escape > 0) if binarize else escape)

        for site, escape_list in self.site_to_escape_list.items():
            self.site_to_average_escape[site] = sum(escape_list) / len(escape_list)

        return self

    def predict(self, sites: list[int]) -> np.ndarray:
        """Predict the average escape score at each antigen site.

        :param sites: A list of sites for which to make a prediction.
        :return: A numpy array containing average escape scores for each antigen site.
        """
        return np.array([self.site_to_average_escape[site] for site in sites])


class MutationModel:
    """A frequency baseline that predicts the average escape score of each wildtype-mutant amino acid substitution."""

    def __init__(self) -> None:
        """Initialize the model."""
        self.wildtype_to_mutation_to_escape_list = defaultdict(lambda: defaultdict(list))
        self.wildtype_to_mutation_to_average_escape = defaultdict(lambda: defaultdict(float))

    def fit(self, wildtypes: list[str], mutations: list[str], escapes: list[float], binarize: bool) -> 'MutationModel':
        """Fit the model by computing the average escape score of each wildtype-mutation amino acid substitution.

        :param wildtypes: A list of wildtype amino acids.
        :param mutations: A list of mutated amino acids.
        :param escapes: A list of escape scores corresponding to the substitution of
                        each wildtype amino acid for each mutant amino acid.
        :param binarize: Whether to binarize the escape into 0 or 1 for non-zero values.
        :return: Returns the fitted model.
        """
        for wildtype, mutation, escape in zip(wildtypes, mutations, escapes):
            self.wildtype_to_mutation_to_escape_list[wildtype][mutation].append(int(escape > 0) if binarize else escape)

        for wildtype, mutation_to_escape_list in self.wildtype_to_mutation_to_escape_list.items():
            for mutation, escape_list in mutation_to_escape_list.items():
                self.wildtype_to_mutation_to_average_escape[wildtype][mutation] = sum(escape_list) / len(escape_list)

        return self

    def predict(self, wildtypes: list[str], mutations: list[str]) -> np.ndarray:
        """Predict the average escape score of each wildtype-mutation amino acid substitution.

        :param wildtypes: A list of wildtype amino acids for which to make a prediction.
        :param mutations: A list of mutated amino acids for which to make a prediction.
        :return: A numpy array containing average escape scores for each wildtype-mutation amino acid substitution.
        """
        return np.array([
            self.wildtype_to_mutation_to_average_escape[wildtype][mutation]
            for wildtype, mutation in zip(wildtypes, mutations)
        ])


def train_and_eval_escape(data: pd.DataFrame) -> tuple:  # TODO: specify return type
    """Train and evaluate a model on predicting escape."""
    # TODO: params docstring

    # Split data


def predict_escape(
        data_path: Path,
        save_dir: Path,
        model_type: Literal['mutation_model', 'site_model', 'likelihood', 'embedding'],
        model_granularity: Literal['per-antibody', 'cross-antibody'],
        task_type: Literal['classification', 'regression'],
        split_type: Literal['mutation', 'site', 'antibody', 'antibody_group'],
        antibody_path: Optional[Path] = None,
        antibody_group_method: Optional[Literal['sequence', 'embedding', 'escape']] = None,
        hidden_layer_sizes: tuple[int, ...] = (100, 100, 100),
        antigen_likelihood_path: Optional[Path] = None,
        embedding_granularity: Optional[Literal['sequence', 'residue']] = None,
        antigen_embeddings_path: Optional[Path] = None,
        antigen_embedding_type: Optional[Literal['mutant', 'difference']] = None,
        use_antibody_embeddings: bool = False,
        antibody_embeddings_path: Optional[Path] = None,
        antibody_embedding_method: Optional[Literal['concatenation', 'attention']] = None,
        split_seed: int = 0,
        model_seed: int = 0,
        verbose: bool = False
) -> None:
    """Train a model to predict antigen escape using ESM2 embeddings."""
    # TODO: params docstring copied from args

    # Validate arguments
    if model_granularity == 'per-antibody':
        assert split_type not in {'antibody', 'antibody_group'}

    if split_type == 'antibody_group':
        assert antibody_path is not None and antibody_group_method is not None
    else:
        assert antibody_path is None and antibody_group_method is None

    if model_type == 'likelihood':
        assert antigen_likelihood_path is not None
    else:
        assert antigen_likelihood_path is None

    if model_type == 'embedding':
        assert antigen_embeddings_path is not None and antigen_embedding_type is not None \
               and embedding_granularity is not None
    else:
        assert antigen_embeddings_path is None and antigen_embedding_type is None \
               and embedding_granularity is None and not use_antibody_embeddings

    if use_antibody_embeddings:
        assert antibody_embeddings_path is not None and antibody_embedding_method is not None
    else:
        assert antibody_embeddings_path is None and antibody_embedding_method is None

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)

    # Load antigen likelihoods
    if antigen_likelihood_path is not None:
        antigen_likelihoods: Optional[dict[str, float]] = torch.load(antigen_likelihood_path)

        print(f'Loaded {len(antigen_likelihoods):,} antigen likelihoods')
    else:
        antigen_likelihoods = None

    # Load antigen embeddings
    if antigen_embeddings_path is not None:
        antigen_embeddings: Optional[dict[str, torch.FloatTensor]] = torch.load(antigen_embeddings_path)

        print(f'Loaded {len(antigen_embeddings):,} antigen embeddings with dimensionality '
              f'{len(next(iter(antigen_embeddings.values())))}\n')
    else:
        antigen_embeddings = None

    # Load antibody embeddings
    if antibody_embeddings_path is not None:
        antibody_embeddings: Optional[dict[str, torch.FloatTensor]] = torch.load(antibody_embeddings_path)

        print(f'Loaded {len(antibody_embeddings):,} antibody embeddings with dimensionality '
              f'{len(next(iter(antibody_embeddings.values())))}\n')
    else:
        antibody_embeddings = None

    # Train and evaluate escape depending on model granularity
    if model_granularity == 'per-antibody':
        all_results, all_models = [], []
        antibodies = sorted(data[ANTIBODY_COLUMN].unique())

        for antibody in tqdm(antibodies, desc='Antibodies'):
            antibody_data = data[data[ANTIBODY_COLUMN] == antibody]
            results, model = train_and_eval_escape(
                data=antibody_data
            )
            all_results.append(results)
            all_models.append(model)

        # TODO: do something with results and models
    elif model_granularity == 'cross-antibody':
        results, model = train_and_eval_escape(
            data=data
        )
        # TODO: do something with results and model
    else:
        raise ValueError(f'Model granularity "{model_granularity}" is not supported.')


if __name__ == '__main__':
    class Args(Tap):
        data_path: Path
        """Path to CSV file containing antibody escape data."""
        save_dir: Path
        """Path to directory where results and models will be saved."""
        model_granularity: Literal['per-antibody', 'cross-antibody']
        """The granularity of the model, either one model per antibody or one model across all antibodies."""
        model_type: Literal['mutation_model', 'site_model', 'likelihood', 'embedding']
        """The type of model to train."""
        task_type: Literal['classification', 'regression']
        """The type of task to perform."""
        split_type: Literal['mutation', 'site', 'antibody', 'antibody_group']
        """The type of data split."""
        antibody_path: Optional[Path] = None
        """Path to a CSV file containing antibody sequences and groups."""
        antibody_group_method: Optional[Literal['sequence', 'embedding', 'escape']] = None
        """The method of grouping antibodies for the antibody_group split type."""
        hidden_layer_sizes: tuple[int, ...] = (100, 100, 100)
        """The sizes of the hidden layers of the MLP model that will be trained."""
        antigen_likelihood_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antigen name to (mutant - wildtype) likelihood."""
        embedding_granularity: Optional[Literal['sequence', 'residue']] = None
        """The granularity of the embeddings, either a sequence average or per-residue embeddings."""
        antigen_embeddings_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antigen name to ESM2 embedding."""
        antigen_embedding_type: Optional[Literal['mutant', 'difference']] = None
        """The type of antigen embedding. mutant: The mutant embedding. difference: mutant - wildtype embedding."""
        use_antibody_embeddings: bool = False
        """Whether to use antibody embeddings in addition to antigen embeddings."""
        antibody_embeddings_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antibody name_chain to ESM2 embedding."""
        antibody_embedding_method: Optional[Literal['concatenation', 'attention']] = None
        """Method of including the antibody embeddings with antigen embeddings."""
        split_seed: int = 0
        """The random seed for splitting the data."""
        model_seed: int = 0
        """The random seed for the model weight initialization."""
        verbose: bool = False
        """Whether to print additional debug information."""


    predict_escape(**Args().parse_args().as_dict())
