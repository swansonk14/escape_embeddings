"""Plots the results of multiple experiments."""
from pathlib import Path
from typing import get_args, Optional

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

LIMITED_MODELS = [
    'Mutation', 'Site', 'RNN', 'Likelihood', 'Antigen Seq Mut', 'Antigen Seq Diff',
    'Antigen Res Mut', 'Antigen Res MutDiff', 'Antigen Res Mut + Antibody', 'Antigen Res Mut Att Antibody'
]
MODEL_ORDER = [
    'Mutation', 'Site', 'RNN', 'Likelihood',
    'Antigen Seq Mut', 'Antigen Seq Diff', 'Antigen Res Mut', 'Antigen Res Diff',
    'Antigen Seq MutDiff', 'Antigen Res MutDiff',
    'Antigen Seq Mut + Antibody', 'Antigen Seq Diff + Antibody',
    'Antigen Res Mut + Antibody', 'Antigen Res Diff + Antibody',
    'Antigen Res Mut Att Antibody'
]
MODEL_NAME_TO_ORDER = {
    model_name: index
    for index, model_name in enumerate(MODEL_ORDER)
}


def row_to_model_name(row: pd.Series, newlines: bool = False) -> str:
    """Convert a row from the results DataFrame to a model name.

    :param row: A row from the results DataFrame.
    :param newlines: Whether to add newlines in the model names.
    :return: The model name of the row.
    """
    whitespace = '\n' if newlines else ' '

    if row.model_type == 'embedding':
        model_name = 'Antigen'

        if row.antigen_embedding_granularity == 'sequence':
            model_name += f'{whitespace}Seq'
        elif row.antigen_embedding_granularity == 'residue':
            model_name += f'{whitespace}Res'
        else:
            breakpoint()
            raise ValueError(f'Antigen embedding granularity "{row.antigen_embedding_granularity}" is not supported.')

        if isinstance(row.antigen_embedding_type, str):
            if row.antigen_embedding_type == 'mutant':
                model_name += ' Mut'
            elif row.antigen_embedding_type == 'difference':
                model_name += ' Diff'
            elif row.antigen_embedding_type == 'mutant_difference':
                model_name += ' MutDiff'
            else:
                raise ValueError(f'Antigen embedding type "{row.antigen_embedding_type}" is not supported.')

        if isinstance(row.antibody_embedding_type, str):
            if row.antibody_embedding_type == 'concatenation':
                model_name += f' +{whitespace}Antibody'
            elif row.antibody_embedding_type == 'attention':
                model_name += f' Att{whitespace}Antibody'
            else:
                raise ValueError(f'Antibody embedding type "{row.antibody_embedding_type}" is not supported.')
    else:
        model_name = row.model_type.title()

    return model_name


def get_means_and_stds(results: pd.DataFrame,
                       metric: str,
                       task_type: TASK_TYPE_OPTIONS,
                       model_granularity: MODEL_GRANULARITY_OPTIONS,
                       split_type: SPLIT_TYPE_OPTIONS,
                       models: Optional[list[str]] = None,
                       newlines: bool = False) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Gets means and stds for metric results in a canonical model order.

    :param results: A DataFrame containing all the experiment results.
    :param metric: The metric to extract means and stds for.
    :param task_type: The task type to limit to.
    :param model_granularity: The model granularity to limit to.
    :param split_type: The split type to limit to.
    :param models: An optional list of models that will be included in the plot. If None, includes all models.
    :param newlines: Whether to add newlines in the model names.
    :return: A tuple containing model names, means, and stds in canonical model order.
    """
    # Limit results to this experimental setting
    experiment_results = results[
        (results['task_type'] == task_type)
        & (results['model_granularity'] == model_granularity)
        & (results['split_type'] == split_type)
        ]

    # Remove likelihood if regression since it isn't trained in that capacity
    if task_type == 'regression':
        experiment_results = experiment_results[experiment_results['model_type'] != 'likelihood']

    # Get model names and results
    model_names = np.array([row_to_model_name(row=row, newlines=newlines) for _, row in experiment_results.iterrows()])
    mean_values = experiment_results[f'{metric}_mean'].to_numpy()
    std_values = experiment_results[f'{metric}_std'].to_numpy()

    # Optionally limit models
    if models is not None:
        model_mask = np.isin([model_name.replace('\n', ' ') for model_name in model_names], models)
        model_names = model_names[model_mask]
        mean_values = mean_values[model_mask]
        std_values = std_values[model_mask]

    # Sort results in canonical order
    argsort = sorted(range(len(model_names)),
                     key=lambda index: MODEL_NAME_TO_ORDER[model_names[index].replace('\n', ' ')])
    model_names = model_names[argsort]
    mean_values = mean_values[argsort]
    std_values = std_values[argsort]

    return model_names, mean_values, std_values


def plot_results_cross_split(results: pd.DataFrame, save_dir: Path, models: Optional[list[str]] = None) -> None:
    """Plot the results of multiple experiments with a single plot across splits.

    :param results: A DataFrame containing all the experiment results.
    :param save_dir: Path to directory where analysis will be saved.
    :param models: An optional list of models that will be included in the plot. If None, includes all models.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot results
    for task_type in get_args(TASK_TYPE_OPTIONS):
        for metric in METRICS:
            plt.clf()

            all_model_names, all_mean_values, all_std_values, all_splits = [], [], [], []

            for split_type in get_args(SPLIT_TYPE_OPTIONS):
                for model_granularity in get_args(MODEL_GRANULARITY_OPTIONS):
                    if model_granularity == 'per-antibody' and split_type in {'antibody', 'antibody_group'}:
                        continue

                    # Get model names, mean values, and std values for this setting
                    model_names, mean_values, std_values = get_means_and_stds(
                        results=results,
                        metric=metric,
                        task_type=task_type,
                        model_granularity=model_granularity,
                        split_type=split_type,
                        models=models,
                        newlines=False
                    )
                    all_model_names.append(model_names)
                    all_mean_values.append(mean_values)
                    all_std_values.append(std_values)
                    all_splits.append(f'{split_type.replace("_", " ").title()}\n{model_granularity.title()}')

            # Ensure all the same model names
            assert all((all_model_names[i] == all_model_names[0]).all() for i in range(len(all_model_names)))

            all_mean_values = np.array(all_mean_values)
            all_std_values = np.array(all_std_values)

            model_names = all_model_names[0]
            num_splits = len(all_model_names)
            num_models = len(model_names)
            width = 1 / (1.25 * num_models)
            offset = width * num_models / 2

            for i in range(num_models):
                model_name = model_names[i]
                mean_values = all_mean_values[:, i]
                std_values = all_std_values[:, i]
                plt.bar(np.arange(num_splits) + i * width - offset, mean_values,
                        width=width, yerr=std_values, alpha=0.5, label=model_name, align='edge',
                        error_kw=dict(lw=1, capsize=2, capthick=1))

            plt.xticks(np.arange(num_splits), all_splits, fontsize=6)
            plt.ylabel(metric)
            plt.title(f'{task_type.title()} {metric}')
            plt.legend(fontsize=6)

            # Save plot
            experiment_save_path = save_dir / task_type / \
                                   f'cross-split-{task_type}-{metric}{"-limited" if models is not None else ""}.pdf'
            experiment_save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(experiment_save_path, bbox_inches='tight')


def plot_results_per_split(results: pd.DataFrame, save_dir: Path, models: Optional[list[str]] = None) -> None:
    """Plot the results of multiple experiments with a separate plot for each split type.

    :param results: A DataFrame containing all the experiment results.
    :param save_dir: Path to directory where analysis will be saved.
    :param models: An optional list of models that will be included in the plot. If None, includes all models.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot results
    for task_type in get_args(TASK_TYPE_OPTIONS):
        for metric in METRICS:
            for split_type in get_args(SPLIT_TYPE_OPTIONS):
                for model_granularity in get_args(MODEL_GRANULARITY_OPTIONS):
                    if model_granularity == 'per-antibody' and split_type in {'antibody', 'antibody_group'}:
                        continue

                    # Get model names, mean values, and std values for this setting
                    model_names, mean_values, std_values = get_means_and_stds(
                        results=results,
                        metric=metric,
                        task_type=task_type,
                        model_granularity=model_granularity,
                        split_type=split_type,
                        models=models,
                        newlines=True
                    )

                    # Plot results
                    plt.clf()
                    plt.bar(model_names, mean_values, alpha=0.5, yerr=std_values, capsize=5)
                    plt.xticks(fontsize=5)
                    plt.ylabel(metric)
                    plt.title(f'{split_type.replace("_", " ").title()} Split {model_granularity.title()} '
                              f'{task_type.title()} {metric}')

                    # Save plot
                    experiment_save_path = save_dir / split_type / model_granularity / task_type / \
                                           f'by-split-{split_type}-{model_granularity}-{task_type}-{metric}' \
                                           f'{"-limited" if models is not None else ""}.pdf'
                    experiment_save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(experiment_save_path, bbox_inches='tight')


def plot_results(results_path: Path, save_dir: Path) -> None:
    """Plots the results of multiple experiments.

    :param results_path: Path to a CSV file containing all the experiment results from combine_results.py.
    :param save_dir: Path to directory where analysis will be saved.
    """
    # Load results
    results = pd.read_csv(results_path)

    # Plot results across splits
    plot_results_cross_split(
        results=results,
        save_dir=save_dir / 'cross_split'
    )

    # Plot results across splits with limited models
    plot_results_cross_split(
        results=results,
        save_dir=save_dir / 'cross_split_limited',
        models=LIMITED_MODELS
    )

    # Plot results by split
    plot_results_per_split(
        results=results,
        save_dir=save_dir / 'by_split'
    )

    # Plot results by split with limited models
    plot_results_per_split(
        results=results,
        save_dir=save_dir / 'by_split_limited',
        models=LIMITED_MODELS
    )


if __name__ == '__main__':
    class Args(Tap):
        results_path: Path  # Path to a CSV file containing all the experiment results from combine_results.py.
        save_dir: Path  # Path to directory where analysis will be saved.


    plot_results(**Args().parse_args().as_dict())
