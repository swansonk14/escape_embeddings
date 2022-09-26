"""Train a model to predict antigen escape using ESM2 embeddings."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tap import Tap
from tqdm import tqdm

from constants import (
    ANTIBODY_COLUMN,
    ANTIBODY_CONDITION_TO_NAME,
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIBODY_NAME_COLUMN,
    ANTIBODY_GROUP_METHOD_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_HIDDEN_LAYER_SIZES,
    EMBEDDING_GRANULARITY_OPTIONS,
    EPITOPE_GROUP_COLUMN,
    ESCAPE_COLUMN,
    MODEL_GRANULARITY_OPTIONS,
    MODEL_TYPE_OPTIONS,
    MUTATION_COLUMN,
    RBD_SEQUENCE,
    SITE_COLUMN,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS,
    WILDTYPE_COLUMN
)


def prune_wildtype_sequences(data: pd.DataFrame) -> pd.DataFrame:
    """Prune wildtype sequences from a DataFrame so that there is only one wildtype seqeunce.

    :param data: DataFrame containing mutation escape data with multiple wildtype sequences.
    :return: A new DataFrame containing mutation escape data with only one wildtype sequence
    """
    # Get mask of wildtype sequences in data
    wildtype_mask = data[WILDTYPE_COLUMN] == data[MUTATION_COLUMN]

    # Ensure one wildtype sequence for each RBD site
    assert sum(wildtype_mask) == len(RBD_SEQUENCE)

    # Ensure all wildtype sequences have zero escape
    assert all(data[wildtype_mask][ESCAPE_COLUMN] == 0)

    # Keep only one copy of wildtype sequence
    first_wildtype_idx = wildtype_mask.idxmax()
    wildtype_mask.iloc[first_wildtype_idx] = False
    data = data[~wildtype_mask]

    return data


def split_data(
        data: pd.DataFrame,
        split_type: SPLIT_TYPE_OPTIONS,
        antibody_path: Optional[Path] = None,
        antibody_group_method: Optional[ANTIBODY_GROUP_METHOD_OPTIONS] = None,
        split_seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test DataFrames."""
    # TODO: params docstring

    if split_type == 'mutation':
        indices = np.arange(len(data))
        train_indices, test_indices = train_test_split(indices, random_state=split_seed, test_size=0.2)
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

    elif split_type == 'site':
        sites = sorted(data[SITE_COLUMN].unique())
        train_sites, test_sites = train_test_split(sites, random_state=split_seed, test_size=0.2)
        train_data = data[data[SITE_COLUMN].isin(train_sites)]
        test_data = data[data[SITE_COLUMN].isin(test_sites)]

    elif split_type == 'antibody':
        antibodies = sorted(data[ANTIBODY_COLUMN].unique())
        train_antibodies, test_antibodies = train_test_split(antibodies, random_state=split_seed, test_size=0.2)
        train_data = data[data[ANTIBODY_COLUMN].isin(train_antibodies)]
        test_data = data[data[ANTIBODY_COLUMN].isin(test_antibodies)]

    elif split_type == 'antibody_group':
        if antibody_group_method == 'sequence':
            raise NotImplementedError

        elif antibody_group_method == 'embedding':
            raise NotImplementedError

        elif antibody_group_method == 'escape':
            # Load antibody data
            antibody_data = pd.read_csv(antibody_path)

            # Get antibody groups
            antibody_name_to_group = dict(zip(antibody_data[ANTIBODY_NAME_COLUMN], antibody_data[EPITOPE_GROUP_COLUMN]))
            data[EPITOPE_GROUP_COLUMN] = [
                antibody_name_to_group[ANTIBODY_CONDITION_TO_NAME.get(antibody, antibody)]
                for antibody in data[ANTIBODY_COLUMN]
            ]

            # Split based on antibody group
            antibody_groups = sorted(data[EPITOPE_GROUP_COLUMN].unique())
            train_antibody_groups, test_antibody_groups = train_test_split(antibody_groups,
                                                                           random_state=split_seed, test_size=0.2)
            train_data = data[data[ANTIBODY_COLUMN].isin(train_antibody_groups)]
            test_data = data[data[ANTIBODY_COLUMN].isin(test_antibody_groups)]
        else:
            raise ValueError(f'Antibody group method "{antibody_group_method}" is not supported.')

    else:
        raise ValueError(f'Split type "{split_type}" is not supported.')

    return train_data, test_data


def train_and_eval_escape(
        data: pd.DataFrame,
        split_type: SPLIT_TYPE_OPTIONS,
        antibody_path: Optional[Path] = None,
        antibody_group_method: Optional[ANTIBODY_GROUP_METHOD_OPTIONS] = None,
        split_seed: int = 0
) -> tuple:  # TODO: specify return type (results, model)
    """Train and evaluate a model on predicting escape."""
    # TODO: params docstring

    print(f'Data size = {len(data)}:,')

    # Prune wildtype sequences to leave only one
    data = prune_wildtype_sequences(data=data)

    print(f'Data size after pruning wildtype sequences = {len(data):,}')

    # Split data
    train_data, test_data = split_data(
        data=data,
        split_type=split_type,
        antibody_path=antibody_path,
        antibody_group_method=antibody_group_method,
        split_seed=split_seed
    )

    print(f'Train size = {len(train_data):,}')
    print(f'Test size = {len(test_data):,}')




def predict_escape(
        data_path: Path,
        save_dir: Path,
        model_granularity: MODEL_GRANULARITY_OPTIONS,
        model_type: MODEL_TYPE_OPTIONS,
        task_type: TASK_TYPE_OPTIONS,
        split_type: SPLIT_TYPE_OPTIONS,
        antibody_path: Optional[Path] = None,
        antibody_group_method: Optional[ANTIBODY_GROUP_METHOD_OPTIONS] = None,
        antigen_likelihood_path: Optional[Path] = None,
        embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antigen_embeddings_path: Optional[Path] = None,
        antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None,
        antibody_embeddings_path: Optional[Path] = None,
        antibody_embedding_method: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
        hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES,
        split_seed: int = 0,
        model_seed: int = 0
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
               and embedding_granularity is None and antibody_embeddings_path is not None

    if antibody_embeddings_path is not None:
        assert antibody_embedding_method is not None
    else:
        assert antibody_embedding_method is None

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
                data=antibody_data,
                split_type=split_type,
                antibody_path=antibody_path,
                antibody_group_method=antibody_group_method,
                split_seed=split_seed
            )
            all_results.append(results)
            all_models.append(model)

        # TODO: do something with results and models
    elif model_granularity == 'cross-antibody':
        results, model = train_and_eval_escape(
            data=data,
            split_type=split_type,
            antibody_path=antibody_path,
            antibody_group_method=antibody_group_method,
            split_seed=split_seed
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
        model_granularity: MODEL_GRANULARITY_OPTIONS
        """The granularity of the model, either one model per antibody or one model across all antibodies."""
        model_type: MODEL_TYPE_OPTIONS
        """The type of model to train."""
        task_type: TASK_TYPE_OPTIONS
        """The type of task to perform."""
        split_type: SPLIT_TYPE_OPTIONS
        """The type of data split."""
        antibody_path: Optional[Path] = None
        """Path to a CSV file containing antibody sequences and groups."""
        antibody_group_method: Optional[ANTIBODY_GROUP_METHOD_OPTIONS] = None
        """The method of grouping antibodies for the antibody_group split type."""
        antigen_likelihood_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antigen name to (mutant - wildtype) likelihood."""
        embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None
        """The granularity of the embeddings, either a sequence average or per-residue embeddings."""
        antigen_embeddings_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antigen name to ESM2 embedding."""
        antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None
        """The type of antigen embedding. mutant: The mutant embedding. difference: mutant - wildtype embedding."""
        antibody_embeddings_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antibody name_chain to ESM2 embedding."""
        antibody_embedding_method: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None
        """Method of including the antibody embeddings with antigen embeddings."""
        hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES
        """The sizes of the hidden layers of the MLP model that will be trained."""
        split_seed: int = 0
        """The random seed for splitting the data."""
        model_seed: int = 0
        """The random seed for the model weight initialization."""


    predict_escape(**Args().parse_args().as_dict())
