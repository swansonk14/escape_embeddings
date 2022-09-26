"""Train a model to predict antigen escape using ESM2 embeddings."""
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import torch
from tap import Tap
from tqdm import tqdm

from constants import (
    ANTIBODY_COLUMN
)


def train_and_eval_escape(data: pd.DataFrame) -> None:
    """Train and evaluate a model on predicting escape."""
    # TODO: params docstring

    pass


def predict_escape(
        data_path: Path,
        save_dir: Path,
        model_type: Literal['baseline_mutation', 'baseline_site', 'likelihood', 'embedding'],
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

    # Load antigen embeddings
    if antigen_embeddings_path is not None:
        antigen_embeddings: dict[str, ] = torch.load(antigen_embeddings_path)

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
        antibodies = sorted(data[ANTIBODY_COLUMN].unique())

        for antibody in tqdm(antibodies, desc='Antibodies'):
            antibody_data = data[data[ANTIBODY_COLUMN] == antibody]
            train_and_eval_escape(
                data=antibody_data
            )

    elif model_granularity == 'cross-antibody':
        train_and_eval_escape(
            data=data
        )
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
        model_type: Literal['baseline_mutation', 'baseline_site', 'likelihood', 'embedding']
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
