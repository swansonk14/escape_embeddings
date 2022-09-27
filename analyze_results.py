"""Analyze the results of multiple experiments."""
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from constants import (
    METRICS,
    MODEL_GRANULARITY_OPTIONS,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS
)

MODEL_ORDER = [
    'Mutation', 'Site', 'Likelihood', 'Antigen Seq Mut', 'Antigen Seq Diff', 'Antigen Res Mut', 'Antigen Res Diff',
    'Antigen Seq Mut Concat Antibody', 'Antigen Seq Diff Concat Antibody',
    'Antigen Res Mut Concat Antibody', 'Antigen Res Diff Concat Antibody',
]
MODEL_NAME_TO_ORDER = {
    model_name: index
    for index, model_name in enumerate(MODEL_ORDER)
}


def row_to_model_name(row: pd.Series) -> str:
    """Convert a row from the results DataFrame to a model name.

    :param row: A row from the results DataFrame.
    :return: The model name of the row.
    """
    if row.model_type == 'embedding':
        model_name = 'Antigen'

        if row.antigen_embedding_granularity == 'sequence':
            model_name += '\nSeq'
        elif row.antigen_embedding_granularity == 'residue':
            model_name += '\nRes'
        else:
            breakpoint()
            raise ValueError(f'Antigen embedding granularity "{row.antigen_embedding_granularity}" is not supported.')

        if isinstance(row.antigen_embedding_type, str):
            if row.antigen_embedding_type == 'mutant':
                model_name += ' Mut'
            elif row.antigen_embedding_type == 'difference':
                model_name += ' Diff'
            else:
                raise ValueError(f'Antigen embedding type "{row.antigen_embedding_type}" is not supported.')

        if isinstance(row.antibody_embedding_type, str):
            if row.antibody_embedding_type == 'concatenation':
                model_name += '\nConcat\nAntibody'
            elif row.antibody_embedding_type == 'attention':
                model_name += '\nAttn\nAntibody'
            else:
                raise ValueError(f'Antibody embedding type "{row.antibody_embedding_type}" is not supported.')
    else:
        model_name = row.model_type.title()

    return model_name


# TODO: need to update for attention and antibody embedding granularity and other antibody group methods
# TODO: likelihood for classification as well
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

                    # Remove likelihood if regression since it isn't trained in that capacity
                    if task_type == 'regression':
                        experiment_results = experiment_results[experiment_results['model_type'] != 'likelihood']

                    # Get model names and results
                    model_names = np.array([row_to_model_name(row) for _, row in experiment_results.iterrows()])
                    metric_values = experiment_results[metric].to_numpy()

                    # Sort results in canonical order
                    argsort = sorted(range(len(model_names)),
                                     key=lambda index: MODEL_NAME_TO_ORDER[model_names[index].replace('\n', ' ')])
                    model_names = model_names[argsort]
                    metric_values = metric_values[argsort]

                    # Plot results
                    plt.clf()
                    plt.bar(model_names, metric_values)
                    plt.xticks(fontsize=5)
                    plt.ylabel(f'{"Mean " if model_granularity == "per-antibody" else ""}{metric}')
                    plt.title(f'{split_type.title()} Split {model_granularity.title()} {task_type.title()} {metric}')

                    # Save plot
                    experiment_save_dir = save_dir / split_type / model_granularity / task_type / f'{metric}.pdf'
                    experiment_save_dir.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(experiment_save_dir, bbox_inches='tight')


if __name__ == '__main__':
    class Args(Tap):
        results_path: Path  # Path to a CSV file containing all the experiment results from combine_results.py.
        save_dir: Path  # Path to directory where analysis will be saved.

    analyze_results(**Args().parse_args().as_dict())
