"""Train a model to predict antigen escape using ESM2 embeddings."""
import json
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from args import PredictEscapeArgs
from constants import (
    ANTIBODY_COLUMN,
    ANTIBODY_CONDITION_TO_NAME,
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIBODY_NAME_COLUMN,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_ATTENTION_NUM_HEADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_HIDDEN_LAYER_DIMS,
    DEFAULT_NUM_EPOCHS_CROSS_ANTIBODY,
    DEFAULT_NUM_EPOCHS_PER_ANTIBODY,
    DEFAULT_RNN_HIDDEN_DIM,
    EMBEDDING_GRANULARITY_OPTIONS,
    EPITOPE_GROUP_COLUMN,
    ESCAPE_COLUMN,
    MODEL_GRANULARITY_OPTIONS,
    MODEL_TYPE_OPTIONS,
    MUTANT_COLUMN,
    NUM_FOLDS,
    SITE_COLUMN,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS,
    WILDTYPE_COLUMN
)
from models import EmbeddingModel, LikelihoodModel, MutationModel, RNNModel, SiteModel


def split_data(
        data: pd.DataFrame,
        fold: int,
        split_type: SPLIT_TYPE_OPTIONS,
        antibody_path: Optional[Path] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test DataFrames."""
    # TODO: params docstring
    if split_type == 'mutation':
        train_indices, test_indices = list(KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0).split(data))[fold]
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

    elif split_type == 'site':
        sites = np.array(sorted(data[SITE_COLUMN].unique()))
        train_indices, test_indices = list(KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0).split(sites))[fold]
        train_sites, test_sites = sites[train_indices], sites[test_indices]
        train_data = data[data[SITE_COLUMN].isin(train_sites)]
        test_data = data[data[SITE_COLUMN].isin(test_sites)]

    elif split_type == 'antibody':
        antibodies = np.array(sorted(data[ANTIBODY_COLUMN].unique()))
        train_indices, test_indices = list(KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0).split(antibodies))[fold]
        train_antibodies, test_antibodies = antibodies[train_indices], antibodies[test_indices]
        train_data = data[data[ANTIBODY_COLUMN].isin(train_antibodies)]
        test_data = data[data[ANTIBODY_COLUMN].isin(test_antibodies)]

    elif split_type == 'antibody_group':
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
        antibody_groups = np.array(sorted(data[EPITOPE_GROUP_COLUMN].unique()))
        train_indices, test_indices = list(KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0).split(antibody_groups))[fold]
        train_antibody_groups, test_antibody_groups = antibody_groups[train_indices], antibody_groups[test_indices]
        train_data = data[data[EPITOPE_GROUP_COLUMN].isin(train_antibody_groups)]
        test_data = data[data[EPITOPE_GROUP_COLUMN].isin(test_antibody_groups)]
    else:
        raise ValueError(f'Split type "{split_type}" is not supported.')

    return train_data, test_data


def train_and_eval_escape(
        data: pd.DataFrame,
        fold: int,
        model_type: MODEL_TYPE_OPTIONS,
        task_type: TASK_TYPE_OPTIONS,
        split_type: SPLIT_TYPE_OPTIONS,
        antibody_path: Optional[Path] = None,
        antigen_likelihoods: Optional[dict[str, float]] = None,
        antigen_embeddings: Optional[dict[str, torch.FloatTensor]] = None,
        antigen_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None,
        antibody_embeddings: Optional[dict[str, torch.FloatTensor]] = None,
        antibody_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
        unique_antibodies: Optional[list[str]] = None,
        hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
        rnn_hidden_dim: int = DEFAULT_RNN_HIDDEN_DIM,
        attention_num_heads: int = DEFAULT_ATTENTION_NUM_HEADS,
        num_epochs: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = DEFAULT_DEVICE,
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
        fold=fold,
        split_type=split_type,
        antibody_path=antibody_path
    )

    if verbose:
        print(f'Train size = {len(train_data):,}')
        print(f'Test size = {len(test_data):,}')

    # Build model
    if model_type == 'mutation':
        model = MutationModel(task_type=task_type)
    elif model_type == 'site':
        model = SiteModel(task_type=task_type)
    elif model_type == 'rnn':
        model = RNNModel(
            task_type=task_type,
            num_epochs=num_epochs,
            hidden_dim=rnn_hidden_dim,
            hidden_layer_dims=hidden_layer_dims,
            batch_size=batch_size,
            device=device
        )
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
            unique_antibodies=unique_antibodies,
            num_epochs=num_epochs,
            batch_size=batch_size,
            hidden_layer_dims=hidden_layer_dims,
            attention_num_heads=attention_num_heads,
            device=device
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
        antigen_likelihoods_path: Optional[Path] = None,
        antigen_embeddings_path: Optional[Path] = None,
        antigen_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None,
        antibody_embeddings_path: Optional[Path] = None,
        antibody_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None,
        antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None,
        hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS,
        rnn_hidden_dim: int = DEFAULT_RNN_HIDDEN_DIM,
        attention_num_heads: int = DEFAULT_ATTENTION_NUM_HEADS,
        num_epochs: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = DEFAULT_DEVICE,
        verbose: bool = False
) -> None:
    """Train a model to predict antigen escape using ESM2 embeddings."""
    # TODO: params docstring copied from args

    # Start timer
    start_time = time()

    # Validate arguments
    # TODO: improve error messages
    if model_granularity == 'per-antibody':
        assert split_type not in {'antibody', 'antibody_group'}

    if split_type == 'antibody_group':
        assert antibody_path is not None
    else:
        assert antibody_path is None

    if model_type == 'likelihood':
        assert antigen_likelihoods_path is not None and task_type == 'classification'
    else:
        assert antigen_likelihoods_path is None

    if model_type == 'embedding':
        assert antigen_embeddings_path is not None and antigen_embedding_type is not None \
               and antigen_embedding_granularity is not None
    else:
        assert antigen_embeddings_path is None and antigen_embedding_type is None \
               and antigen_embedding_granularity is None and antibody_embeddings_path is None and num_epochs is None

    if antibody_embeddings_path is not None:
        assert antibody_embedding_type is not None and antibody_embedding_type != 'one-hot' \
               and antibody_embedding_granularity is not None
    else:
        assert antibody_embedding_type in {None, 'one-hot'} and antibody_embedding_granularity is None

    if antibody_embedding_granularity is not None and antibody_embedding_granularity != 'sequence':
        raise NotImplementedError(f'Antibody embedding granularity "{antibody_embedding_granularity}" '
                                  f'has not been implemented yet.')

    if args.antigen_embedding_type == 'linker':
        assert args.antigen_embedding_granularity == 'sequence' and args.antibody_embedding_type is None

    if model_type in {'rnn', 'embedding'} and num_epochs is None:
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
              f'{next(iter(antigen_embeddings.values())).shape[-1]:,}\n')
    else:
        antigen_embeddings = None

    # Load antibody embeddings
    if antibody_embeddings_path is not None:
        antibody_embeddings: Optional[dict[str, torch.FloatTensor]] = torch.load(antibody_embeddings_path)

        print(f'Loaded {len(antibody_embeddings):,} antibody embeddings with dimensionality '
              f'{next(iter(antibody_embeddings.values())).shape[-1]:,}\n')
    else:
        antibody_embeddings = None

    # Get unique antibodies
    unique_antibodies = sorted(set(data[ANTIBODY_COLUMN].unique()))

    # Set up folds and antibodies depending on model granularity
    if model_granularity == 'per-antibody':
        folds = [0]
        antibody_sets = [{antibody} for antibody in unique_antibodies]
    elif model_granularity == 'cross-antibody':
        folds = list(range(NUM_FOLDS))
        antibody_sets = [unique_antibodies]
    else:
        raise ValueError(f'Model granularity "{model_granularity}" is not supported.')

    # Train and evaluate escape
    results = [
        train_and_eval_escape(
            data=data[data[ANTIBODY_COLUMN].isin(antibody_set)],
            fold=fold,
            model_type=model_type,
            task_type=task_type,
            split_type=split_type,
            antibody_path=antibody_path,
            antigen_likelihoods=antigen_likelihoods,
            antigen_embeddings=antigen_embeddings,
            antigen_embedding_granularity=antigen_embedding_granularity,
            antigen_embedding_type=antigen_embedding_type,
            antibody_embeddings=antibody_embeddings,
            antibody_embedding_granularity=antibody_embedding_granularity,
            antibody_embedding_type=antibody_embedding_type,
            unique_antibodies=unique_antibodies,
            hidden_layer_dims=hidden_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            attention_num_heads=attention_num_heads,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            verbose=verbose
        )
        for fold in tqdm(folds, desc='Folds')
        for antibody_set in tqdm(antibody_sets, desc='Antibodies', leave=False)
    ]

    # Compute summary results
    metrics = sorted(results[0].keys())
    all_results = {
        metric: [result[metric] for result in results]
        for metric in metrics
    }
    summary_results = {
        metric: {
            'mean': float(np.nanmean(all_results[metric])),
            'std': float(np.nanstd(all_results[metric])),
            'num': len(all_results[metric]),
            'num_nan': int(np.isnan(all_results[metric]).sum()),
            'values': all_results[metric],
            'items': unique_antibodies if model_granularity == 'per-antibody' else [f'fold_{fold}' for fold in folds]
        }
        for metric in metrics
    }

    for metric, values in summary_results.items():
        print(f'Test {metric} = {values["mean"]:.3f} +/- {values["std"]:.3f} ' +
              (f'({values["num_nan"]:,} / {values["num"]:,} NaN)' if values["num_nan"] > 0 else ''))

    # End timer
    summary_results['time'] = time() - start_time

    print(f'Total time = {timedelta(seconds=summary_results["time"])}')

    # Save summary results
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(summary_results, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    # Parse args
    args = PredictEscapeArgs().parse_args()

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
