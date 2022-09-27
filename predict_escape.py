"""Train a model to predict antigen escape using ESM2 embeddings."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error, r2_score
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
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_LAYER_DIMS,
    DEFAULT_NUM_EPOCHS_CROSS_ANTIBODY,
    DEFAULT_NUM_EPOCHS_PER_ANTIBODY,
    EMBEDDING_GRANULARITY_OPTIONS,
    EPITOPE_GROUP_COLUMN,
    ESCAPE_COLUMN,
    MODEL_GRANULARITY_OPTIONS,
    MODEL_TYPE_OPTIONS,
    MUTANT_COLUMN,
    SITE_COLUMN,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS,
    WILDTYPE_COLUMN
)
from models import EmbeddingModel, LikelihoodModel, MutationModel, SiteModel


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
            data = data.copy()
            data[EPITOPE_GROUP_COLUMN] = [
                antibody_name_to_group[antibody]
                for antibody in data[ANTIBODY_COLUMN]
            ]

            # Split based on antibody group
            antibody_groups = sorted(data[EPITOPE_GROUP_COLUMN].unique())
            train_antibody_groups, test_antibody_groups = train_test_split(antibody_groups,
                                                                           random_state=split_seed, test_size=0.2)
            train_data = data[data[EPITOPE_GROUP_COLUMN].isin(train_antibody_groups)]
            test_data = data[data[EPITOPE_GROUP_COLUMN].isin(test_antibody_groups)]
        else:
            raise ValueError(f'Antibody group method "{antibody_group_method}" is not supported.')

    else:
        raise ValueError(f'Split type "{split_type}" is not supported.')

    return train_data, test_data


def train_and_eval_escape(
        data: pd.DataFrame,
        model_type: MODEL_TYPE_OPTIONS,
        task_type: TASK_TYPE_OPTIONS,
        split_type: SPLIT_TYPE_OPTIONS,
        antibody_path: Optional[Path] = None,
        antibody_group_method: Optional[ANTIBODY_GROUP_METHOD_OPTIONS] = None,
        antigen_likelihoods: Optional[dict[str, float]] = None,
        antigen_embeddings: Optional[dict[str, torch.FloatTensor]] = None,
        antigen_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None,
        antibody_embeddings: Optional[dict[str, torch.FloatTensor]] = None,
        antibody_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
        hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
        num_epochs: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        split_seed: int = 0,
        model_seed: int = 0,
        verbose: bool = True
) -> dict[str, Optional[float]]:
    """Train and evaluate a model on predicting escape.

    TODO: params docstring
    :return: A results dictionary mapping metric to value.
    """
    if verbose:
        print(f'Data size = {len(data):,}')

    # Remove wildtype sequences
    data = data[data[WILDTYPE_COLUMN] != data[MUTANT_COLUMN]]

    if verbose:
        print(f'Data size after removing wildtype sequences = {len(data):,}')

    # Split data
    train_data, test_data = split_data(
        data=data,
        split_type=split_type,
        antibody_path=antibody_path,
        antibody_group_method=antibody_group_method,
        split_seed=split_seed
    )

    if verbose:
        print(f'Train size = {len(train_data):,}')
        print(f'Test size = {len(test_data):,}')

    # Build model
    if model_type == 'mutation':
        model = MutationModel(task_type=task_type)
    elif model_type == 'site':
        model = SiteModel(task_type=task_type)
    elif model_type == 'likelihood':
        model = LikelihoodModel(task_type=task_type, antigen_likelihoods=antigen_likelihoods)
    elif model_type == 'embedding':
        model = EmbeddingModel(
            task_type=task_type,
            antigen_embeddings=antigen_embeddings,
            antigen_embedding_granularity=antigen_embedding_granularity,
            antigen_embedding_type=antigen_embedding_type,
            antibody_embeddings=antibody_embeddings,
            antibody_embedding_granularity=antibody_embedding_granularity,
            antibody_embedding_type=antibody_embedding_type,
            num_epochs=num_epochs,
            batch_size=batch_size,
            hidden_layer_dims=hidden_layer_dims,
            model_seed=model_seed
        )
    else:
        raise ValueError(f'Model type "{model_type}" is not supported.')

    if verbose:
        print(f'Model = {model}')

    # Train model
    model.fit(
        antibodies=list(train_data[ANTIBODY_COLUMN]),
        sites=list(train_data[SITE_COLUMN]),
        wildtypes=list(train_data[WILDTYPE_COLUMN]),
        mutants=list(train_data[MUTANT_COLUMN]),
        escapes=list(train_data[ESCAPE_COLUMN])
    )

    # Make predictions
    test_preds = model.predict(
        antibodies=list(test_data[ANTIBODY_COLUMN]),
        sites=list(test_data[SITE_COLUMN]),
        wildtypes=list(test_data[WILDTYPE_COLUMN]),
        mutants=list(test_data[MUTANT_COLUMN])
    )

    # Binarize test escape scores
    test_escape = test_data[ESCAPE_COLUMN].to_numpy()
    binary_test_escape = (test_escape > 0).astype(int)
    unique_binary_labels = set(binary_test_escape)

    # Evaluate predictions
    # TODO: average by antibody or across mutations?
    results = {
        'ROC-AUC': roc_auc_score(binary_test_escape, test_preds) if len(unique_binary_labels) > 1 else float('nan'),
        'PRC-AUC': average_precision_score(binary_test_escape, test_preds) if len(unique_binary_labels) > 1 else float('nan'),
        'MSE': mean_squared_error(test_escape, test_preds),
        'R2': r2_score(test_escape, test_preds)
    }

    if verbose:
        for metric, value in results.items():
            print(f'Test {metric} = {value:.3f}')

    return results


def predict_escape(
        data_path: Path,
        save_dir: Path,
        model_granularity: MODEL_GRANULARITY_OPTIONS,
        model_type: MODEL_TYPE_OPTIONS,
        split_type: SPLIT_TYPE_OPTIONS,
        task_type: TASK_TYPE_OPTIONS,
        antibody_path: Optional[Path] = None,
        antibody_group_method: Optional[ANTIBODY_GROUP_METHOD_OPTIONS] = None,
        antigen_likelihoods_path: Optional[Path] = None,
        antigen_embeddings_path: Optional[Path] = None,
        antigen_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None,
        antibody_embeddings_path: Optional[Path] = None,
        antibody_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
        hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
        num_epochs: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        split_seed: int = 0,
        model_seed: int = 0
) -> None:
    """Train a model to predict antigen escape using ESM2 embeddings."""
    # TODO: params docstring copied from args

    # Validate arguments
    # TODO: improve error messages
    if model_granularity == 'per-antibody':
        assert split_type not in {'antibody', 'antibody_group'}

    if split_type == 'antibody_group':
        assert antibody_path is not None and antibody_group_method is not None
    else:
        assert antibody_path is None and antibody_group_method is None

    if model_type == 'likelihood':
        assert antigen_likelihoods_path is not None and task_type == 'regression'
    else:
        assert antigen_likelihoods_path is None

    if model_type == 'embedding':
        assert antigen_embeddings_path is not None and antigen_embedding_type is not None \
               and antigen_embedding_granularity is not None
    else:
        assert antigen_embeddings_path is None and antigen_embedding_type is None \
               and antigen_embedding_granularity is None and antibody_embeddings_path is None and num_epochs is None

    if antibody_embeddings_path is not None:
        assert antibody_embedding_type is not None and antibody_embedding_granularity is not None
    else:
        assert antibody_embedding_type is None and antibody_embedding_granularity is None

    if antibody_embedding_granularity == 'residue':
        assert antibody_embedding_type == 'attention'

    if model_type == 'embedding' and num_epochs is None:
        if model_granularity == 'per-antibody':
            num_epochs = DEFAULT_NUM_EPOCHS_PER_ANTIBODY
        elif model_granularity == 'cross-antibody':
            num_epochs = DEFAULT_NUM_EPOCHS_CROSS_ANTIBODY
        else:
            raise ValueError(f'Model granularity "{model_granularity}" is not supported.')

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)

    # Correct antibody names to match antibody sequence data and embeddings
    data[ANTIBODY_COLUMN] = [ANTIBODY_CONDITION_TO_NAME.get(antibody, antibody) for antibody in data[ANTIBODY_COLUMN]]

    # Load antigen likelihoods
    if antigen_likelihoods_path is not None:
        antigen_likelihoods: Optional[dict[str, float]] = torch.load(antigen_likelihoods_path)

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
        all_results = []
        antibodies = sorted(data[ANTIBODY_COLUMN].unique())

        for antibody in tqdm(antibodies, desc='Antibodies'):
            antibody_data = data[data[ANTIBODY_COLUMN] == antibody]
            results = train_and_eval_escape(
                data=antibody_data,
                model_type=model_type,
                task_type=task_type,
                split_type=split_type,
                antibody_path=antibody_path,
                antibody_group_method=antibody_group_method,
                antigen_likelihoods=antigen_likelihoods,
                antigen_embeddings=antigen_embeddings,
                antigen_embedding_granularity=antigen_embedding_granularity,
                antigen_embedding_type=antigen_embedding_type,
                antibody_embeddings=antibody_embeddings,
                antibody_embedding_granularity=antibody_embedding_granularity,
                antibody_embedding_type=antibody_embedding_type,
                hidden_layer_dims=hidden_layer_dims,
                num_epochs=num_epochs,
                batch_size=batch_size,
                split_seed=split_seed,
                model_seed=model_seed,
                verbose=False
            )
            all_results.append(results)

            # Create antibody save dir
            antibody_save_dir = save_dir / antibody
            antibody_save_dir.mkdir(parents=True, exist_ok=True)

            # Save results
            with open(antibody_save_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=4, sort_keys=True)

        # Compute summary results
        metrics = all_results[0].keys()
        all_results = {
            metric: [result[metric] for result in all_results]
            for metric in metrics
        }
        summary_results = {
            metric: {
                'mean': float(np.nanmean(all_results[metric])),
                'std': float(np.nanstd(all_results[metric])),
                'num': len(all_results[metric]),
                'num_nan': int(np.isnan(all_results[metric]).sum())
            }
            for metric in metrics
        }

        for metric, values in summary_results.items():
            print(f'Test {metric} = {values["mean"]:.3f} +/- {values["std"]:.3f} ' +
                  (f'({values["num_nan"]:,} / {values["num"]:,} NaN)' if values["num_nan"] > 0 else ''))

        # Save summary results
        with open(save_dir / 'summary_results.json', 'w') as f:
            json.dump(summary_results, f, indent=4, sort_keys=True)

    elif model_granularity == 'cross-antibody':
        results = train_and_eval_escape(
            data=data,
            model_type=model_type,
            task_type=task_type,
            split_type=split_type,
            antibody_path=antibody_path,
            antibody_group_method=antibody_group_method,
            antigen_likelihoods=antigen_likelihoods,
            antigen_embeddings=antigen_embeddings,
            antigen_embedding_granularity=antigen_embedding_granularity,
            antigen_embedding_type=antigen_embedding_type,
            antibody_embeddings=antibody_embeddings,
            antibody_embedding_granularity=antibody_embedding_granularity,
            antibody_embedding_type=antibody_embedding_type,
            hidden_layer_dims=hidden_layer_dims,
            num_epochs=num_epochs,
            batch_size=batch_size,
            split_seed=split_seed,
            model_seed=model_seed,
            verbose=True
        )

        # Save results
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4, sort_keys=True)
    else:
        raise ValueError(f'Model granularity "{model_granularity}" is not supported.')


if __name__ == '__main__':
    class Args(Tap):
        data_path: Path
        """Path to CSV file containing antibody escape data."""
        save_dir: Path
        """Path to directory where results will be saved."""
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
        antigen_likelihoods_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antigen name to (mutant - wildtype) likelihood."""
        antigen_embeddings_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antigen name to ESM2 embedding."""
        antigen_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None
        """The granularity of the antigen embeddings, either a sequence average or per-residue embeddings."""
        antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None
        """The type of antigen embedding. mutant: The mutant embedding. difference: mutant - wildtype embedding."""
        antibody_embeddings_path: Optional[Path] = None
        """Path to PT file containing a dictionary mapping from antibody name_chain to ESM2 embedding."""
        antibody_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None
        """The granularity of the antibody embeddings, either a sequence average or per-residue embeddings."""
        antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None
        """Method of including the antibody embeddings with antigen embeddings."""
        hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS
        """The sizes of the hidden layers of the MLP model that will be trained."""
        num_epochs: Optional[int] = None
        """The number of epochs for the embedding model. If None, num_epochs is set based on model_granularity."""
        batch_size: int = DEFAULT_BATCH_SIZE
        """The batch size for the embedding model."""
        split_seed: int = 0
        """The random seed for splitting the data."""
        model_seed: int = 0
        """The random seed for the model weight initialization."""
        skip_existing: bool = False
        """Whether to skip running the code if the save_dir already exists."""

    # Parse args
    args = Args().parse_args()

    # Skip existing
    if args.skip_existing and args.save_dir.exists():
        print('skip_existing is True and save_dir already exists. Exiting...')
        exit()

    # Save args
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.save(args.save_dir / 'args.json')

    # Prepare args dict
    args_dict = args.as_dict()
    del args_dict['skip_existing']

    # Predict escape
    predict_escape(**args_dict)
