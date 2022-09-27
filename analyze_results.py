"""Analyze the results of multiple experiments."""
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
from tap import Tap

from constants import (
    METRICS,
    MODEL_GRANULARITY_OPTIONS,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS
)


def row_to_model_name(row: pd.Series) -> str:
    """Convert a row from the results DataFrame to a model name.

    :param row: A row from the results DataFrame.
    :return: The model name of the row.
    """
    model_name = row.model_type.title()

    if row.model_type == 'embedding':
        breakpoint()
        if row.antigen_embedding_granularity == 'sequence':
            model_name += ' Seq'
        elif row.antigen_embedding_granularity == 'residue':
            model_name += ' Res'
        else:
            raise ValueError(f'Antigen embedding granularity "{row.antigen_embedding_granularity}" is not supported.')

        if not np.isnan(row.antigen_embedding_type):
            if row.antigen_embedding_type == 'mutant':
                model_name += ' Mut'
            elif row.antigen_embedding_type == 'difference':
                model_name += ' Diff'
            else:
                raise ValueError(f'Antigen embedding type "{row.antigen_embedding_type}" is not supported.')

        if row.antibody_emedding_type == 'concatenation':
            model_name += ' Concat Antibody'
        elif row.antibody_emedding_type == 'attention':
            model_name += ' Attn Antibody'
        else:
            raise ValueError(f'Antibody embedding type "{row.antibody_embedding_type}" is not supported.')

    return model_name


# TODO: need to update for attention and antibody embedding granularity and other antibody group methods
def analyze_results(results_path: Path, save_dir: Path) -> None:
    """Analyze the results of multiple experiments.

    :param results_path: Path to a CSV file containing all the experiment results from combine_results.py.
    :param save_dir: Path to directory where analysis will be saved.
    """
    # Load results
    results = pd.read_csv(results_path)

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Per-antibody results
    for model_granularity in get_args(MODEL_GRANULARITY_OPTIONS):
        for split_type in get_args(SPLIT_TYPE_OPTIONS):
            if model_granularity == 'per-antibody' and split_type in {'antibody', 'antibody_group'}:
                continue

            for task_type in get_args(TASK_TYPE_OPTIONS):
                for metric in METRICS:
                    # Limit results to this experimental setting
                    experiment_results = results[
                        (results['model_granularity'] == model_granularity)
                        & (results['split_type'] == split_type)
                        & (results['task_type'] == task_type)
                    ]

                    # Get model names and results
                    model_names = [row_to_model_name(row) for _, row in experiment_results.iterrows()]
                    metric_values = experiment_results[metric]

                    breakpoint()


if __name__ == '__main__':
    class Args(Tap):
        results_path: Path  # Path to a CSV file containing all the experiment results from combine_results.py.
        save_dir: Path  # Path to directory where analysis will be saved.

    analyze_results(**Args().parse_args().as_dict())
