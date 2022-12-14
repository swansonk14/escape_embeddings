"""Sets up all experiments for predicting escape with different data, models, and settings."""
from pathlib import Path
from typing import Literal, get_args, Optional

from tap import Tap

from constants import (
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    EMBEDDING_GRANULARITY_OPTIONS,
    MODEL_GRANULARITY_OPTIONS,
    MODEL_TYPE_OPTIONS,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS
)


def setup_experiments(
        data_path: str,
        antibody_path: str,
        antigen_likelihoods_path: str,
        antigen_embeddings_path: str,
        antibody_embeddings_path: str,
        antibody_antigen_embeddings_path: str,
        experiment_save_dir: Path,
        bash_save_path: Path,
        skip_existing: bool = False,
        save_only_device: Optional[str] = None,
        start_index: int = 0
) -> None:
    """Sets up all experiments for predicting escape with different data, models, and settings.

    :param data_path: Path to CSV file containing antibody escape data.
    :param antibody_path: Path to a CSV file containing antibody sequences and groups.
    :param antigen_likelihoods_path: Path to PT file containing a dictionary mapping from antigen name to (mutant - wildtype) likelihood.
    :param antigen_embeddings_path: Path to PT file containing a dictionary mapping from antigen name to ESM2 embedding.
    :param antibody_embeddings_path: Path to PT file containing a dictionary mapping from antibody name_chain to ESM2 embedding.
    :param antibody_antigen_embeddings_path: Path to PT file containing a dictionary mapping from antibody name_chain and antigen name to ESM2 embedding.
    :param experiment_save_dir: Path to directory where all the experiment results will be saved.
    :param bash_save_path: Path to bash file where experiment commands will be saved.
    :param skip_existing: Whether to skip running the code if the save_dir already exists.
    :param save_only_device: Whether to only save experiments that are run on a certain device (e.g., CPU or GPU).
    :param start_index: Starting index of the experiments to run.
    """
    # Set up experiment args
    all_experiments_args = []
    for model_type in get_args(MODEL_TYPE_OPTIONS):
        for model_granularity in get_args(MODEL_GRANULARITY_OPTIONS):
            for task_type in get_args(TASK_TYPE_OPTIONS):
                for split_type in get_args(SPLIT_TYPE_OPTIONS):
                    if model_granularity == 'per-antibody' and split_type in {'antibody', 'antibody_group'}:
                        continue

                    if model_type == 'likelihood' and task_type != 'classification':
                        continue

                    experiment_args = {
                        'model_type': model_type,
                        'model_granularity': model_granularity,
                        'task_type': task_type,
                        'split_type': split_type
                    }

                    if split_type == 'antibody_group':
                        experiment_args['antibody_path'] = antibody_path

                    if model_type == 'likelihood':
                        experiment_args['antigen_likelihoods_path'] = antigen_likelihoods_path

                    if model_type == 'rnn':
                        experiments_args = []
                        for antigen_embedding_granularity in get_args(EMBEDDING_GRANULARITY_OPTIONS):
                            experiments_args.append(experiment_args | {
                                'antigen_embedding_granularity': antigen_embedding_granularity,
                                'device': 'cuda'
                            })
                    elif model_type == 'embedding':
                        antigen_experiment_args = []
                        for antigen_embedding_granularity in get_args(EMBEDDING_GRANULARITY_OPTIONS):
                            for antigen_embedding_type in get_args(ANTIGEN_EMBEDDING_TYPE_OPTIONS):
                                if antigen_embedding_type == 'linker':
                                    if antigen_embedding_granularity != 'sequence':
                                        continue

                                    antigen_experiment_args.append(experiment_args | {
                                        'antigen_embedding_granularity': antigen_embedding_granularity,
                                        'antigen_embedding_type': antigen_embedding_type,
                                        'antigen_embeddings_path': antibody_antigen_embeddings_path
                                    })
                                else:
                                    antigen_experiment_args.append(experiment_args | {
                                        'antigen_embedding_granularity': antigen_embedding_granularity,
                                        'antigen_embedding_type': antigen_embedding_type,
                                        'antigen_embeddings_path': antigen_embeddings_path
                                    })

                        antibody_experiments_args = []
                        for experiment_args in antigen_experiment_args:
                            for antibody_embedding_granularity in get_args(EMBEDDING_GRANULARITY_OPTIONS):
                                for antibody_embedding_type in get_args(ANTIBODY_EMBEDDING_TYPE_OPTIONS):
                                    # Only do antibody experiments with antigen res mut embeddings
                                    if experiment_args['antigen_embedding_granularity'] != 'residue' or \
                                            experiment_args['antigen_embedding_type'] != 'mutant':
                                        continue

                                    if antibody_embedding_type in {'concatenation', 'one_hot'} \
                                            and antibody_embedding_granularity != 'sequence':
                                        continue

                                    if antibody_embedding_type == 'attention' and (
                                            task_type != 'classification' or antibody_embedding_granularity != 'residue'
                                    ):
                                        continue

                                    antibody_experiments_args.append(experiment_args | {
                                        'antibody_embedding_type': antibody_embedding_type
                                    } | ({'antibody_embeddings_path': antibody_embeddings_path,
                                          'antibody_embedding_granularity': antibody_embedding_granularity}
                                         if antibody_embedding_type != 'one_hot' else {})
                                      | ({'device': 'cuda'} if antibody_embedding_type == 'attention' else {})
                                    )

                        experiments_args = antigen_experiment_args + antibody_experiments_args
                    else:
                        experiments_args = [experiment_args]

                    all_experiments_args += experiments_args

    # Optionally filter out experiments with wrong device
    if save_only_device is not None:
        all_experiments_args = [args for args in all_experiments_args if args.get('device', 'cpu') == save_only_device]

    # Add save directory
    for i, experiment_args in enumerate(all_experiments_args):
        experiment_args['save_dir'] = str(experiment_save_dir / str(i + start_index))

    print(f'Number of experiments = {len(all_experiments_args):,}')

    # Save experiment commands in the correct format
    with open(bash_save_path, 'w') as f:
        f.write('#!/bin/bash\n\n')

        for experiment_args in all_experiments_args:
            args = {'data_path': data_path} | experiment_args

            f.write('python predict_escape.py \\\n')

            keys = sorted(args.keys())
            for i in range(len(keys) - 1):
                f.write(f'    --{keys[i]} {args[keys[i]]} \\\n')

            f.write(f'    --{keys[-1]} {args[keys[-1]]}')

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
        antibody_antigen_embeddings_path: str
        """Path to PT file containing a dictionary mapping from antibody name_chain and antigen name to ESM2 embedding."""
        experiment_save_dir: Path
        """Path to directory where all the experiment results will be saved."""
        bash_save_path: Path
        """Path to bash file where experiment commands will be saved."""
        skip_existing: bool = False
        """Whether to skip running the code if the save_dir already exists."""
        save_only_device: Optional[Literal['cpu', 'cuda']] = None
        """Whether to only save experiments that are run on a certain device (e.g., CPU or GPU)."""
        start_index: int = 0
        """Starting index of the experiments to run."""


    setup_experiments(**Args().parse_args().as_dict())
