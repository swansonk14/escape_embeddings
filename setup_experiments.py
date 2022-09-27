"""Run all experiments of predicting escape with different models and settings."""
from pathlib import Path
from typing import get_args

from tap import Tap

from constants import (
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIBODY_GROUP_METHOD_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    EMBEDDING_GRANULARITY_OPTIONS,
    MODEL_GRANULARITY_OPTIONS,
    MODEL_TYPE_OPTIONS,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS
)


def run_experiments(
        data_path: str,
        antibody_path: str,
        antigen_likelihoods_path: str,
        antigen_embeddings_path: str,
        antibody_embeddings_path: str,
        experiment_save_dir: Path,
        bash_save_path: Path,
        skip_existing: bool
) -> None:
    """Run all experiments of predicting escape with different models and settings.
    TODO: params docstring
    """
    # Set up experiment args
    all_experiments_args = []
    for model_type in get_args(MODEL_TYPE_OPTIONS):
        for model_granularity in get_args(MODEL_GRANULARITY_OPTIONS):
            for task_type in get_args(TASK_TYPE_OPTIONS):
                for split_type in get_args(SPLIT_TYPE_OPTIONS):
                    if model_granularity == 'per-antibody' and split_type in {'antibody', 'antibody_group'}:
                        continue

                    if model_type == 'likelihood' and task_type == 'regression':
                        continue

                    experiment_args = [
                        '--model_type', model_type,
                        '--model_granularity', model_granularity,
                        '--task_type', task_type,
                        '--split_type', split_type
                    ]

                    if model_type == 'likelihood':
                        experiment_args += ['--antigen_likelihoods_path', antigen_likelihoods_path]

                    if split_type == 'antibody_group':
                        experiments_args = [
                            experiment_args + [
                                '--antibody_group_method', antibody_group_method, '--antibody_path', antibody_path
                            ]
                            for antibody_group_method in get_args(ANTIBODY_GROUP_METHOD_OPTIONS)
                            if antibody_group_method == 'escape'  # TODO: remove this
                        ]
                    else:
                        experiments_args = [experiment_args]

                    if model_type == 'embedding':
                        for experiment_args in experiments_args:
                            experiment_args += ['--antigen_embeddings_path', antigen_embeddings_path]

                        antigen_experiment_args = []
                        for experiment_args in experiments_args:
                            for antigen_embedding_granularity in get_args(EMBEDDING_GRANULARITY_OPTIONS):
                                for antigen_embedding_type in get_args(ANTIGEN_EMBEDDING_TYPE_OPTIONS):
                                    antigen_experiment_args.append(experiment_args + [
                                        '--antigen_embedding_granularity', antigen_embedding_granularity,
                                        '--antigen_embedding_type', antigen_embedding_type
                                    ])
                        experiments_args = antigen_experiment_args

                        antibody_experiments_args = []
                        for experiment_args in experiments_args:
                            for antibody_embedding_granularity in get_args(EMBEDDING_GRANULARITY_OPTIONS):
                                for antibody_embedding_type in get_args(ANTIBODY_EMBEDDING_TYPE_OPTIONS):
                                    if antibody_embedding_granularity == 'residue' \
                                            and antibody_embedding_type == 'concatenation':
                                        continue

                                    # TODO: remove this
                                    if antibody_embedding_type == 'attention':
                                        continue

                                    antibody_experiments_args.append(experiment_args + [
                                        '--antibody_embedding_granularity', antibody_embedding_granularity,
                                        '--antibody_embedding_type', antibody_embedding_type,
                                        '--antibody_embeddings_path', antibody_embeddings_path
                                    ])
                        experiments_args += antibody_experiments_args

                    all_experiments_args += experiments_args

    # Add save directory
    for i, experiment_args in enumerate(all_experiments_args):
        experiment_args += ['--save_dir', str(experiment_save_dir / str(i))]

    print(f'Number of experiments = {len(all_experiments_args):,}')

    # Save experiment commands
    with open(bash_save_path, 'w') as f:
        f.write('#!/bin/bash\n\n')

        for experiment_args in all_experiments_args:
            args = ['--data_path', data_path] + experiment_args

            f.write('python predict_escape.py \\\n')

            for i in range(0, len(args) - 2, 2):
                f.write(f'    {args[i]} {args[i + 1]} \\\n')

            f.write(f'    {args[-2]} {args[-1]}')

            if skip_existing:
                f.write(' \\\n    --skip_existing')

            f.write('\n\n')


if __name__ == '__main__':
    class Args(Tap):
        data_path: str
        """Path to CSV file containing antibody escape data."""
        antibody_path: str
        """Path to a CSV file containing antibody sequences and groups."""
        antigen_likelihoods_path: str
        """Path to PT file containing a dictionary mapping from antigen name to (mutant - wildtype) likelihood."""
        antigen_embeddings_path: str
        """Path to PT file containing a dictionary mapping from antigen name to ESM2 embedding."""
        antibody_embeddings_path: str
        """Path to PT file containing a dictionary mapping from antibody name_chain to ESM2 embedding."""
        experiment_save_dir: Path
        """Path to directory where all the experiment results will be saved."""
        bash_save_path: Path
        """Path to bash file where experiment commands will be saved."""
        skip_existing: bool = False
        """Whether to skip running the code if the save_dir already exists."""

    run_experiments(**Args().parse_args().as_dict())
