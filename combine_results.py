"""Combine the results of multiple experiments into a single CSV file."""
import json
from pathlib import Path

import pandas as pd
from tap import Tap

from args import PredictEscapeArgs


def combine_results(results_dir: Path, save_path: Path) -> None:
    """Combine the results of multiple experiments into a single CSV file.

    :param results_dir: Path to directory containing experiment results.
    :param save_path: Path to CSV file where combined results will be saved.
    """
    # Get all results
    results_dicts = []
    results_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    for experiment_results_dir in results_dirs:
        # Get args from experiment
        args = PredictEscapeArgs()
        args.load(experiment_results_dir / 'args.json')

        # Get results from experiment
        with open(experiment_results_dir / 'results.json') as f:
            results = json.load(f)

        # Combine experiment results with args
        results_dicts.append({
            'model_granularity': args.model_granularity,
            'model_type': args.model_type,
            'task_type': args.task_type,
            'split_type': args.split_type,
            'antigen_embedding_granularity': args.antigen_embedding_granularity,
            'antigen_embedding_type': args.antigen_embedding_type,
            'antibody_embedding_granularity': args.antibody_embedding_granularity,
            'antibody_embedding_type': args.antibody_embedding_type,
            **{
                f'{metric}_{value_type}': value_type_to_values[value_type]
                for metric, value_type_to_values in results.items()
                for value_type in ['mean', 'std', 'num', 'num_nan']
                if metric != 'time'
            }
        })

    # Combine all results
    all_results = pd.DataFrame(results_dicts)

    # Save results
    save_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(save_path, index=False)


if __name__ == '__main__':
    class Args(Tap):
        results_dir: Path  # Path to directory containing experiment results.
        save_path: Path  # Path to CSV file where combined results will be saved.

    combine_results(**Args().parse_args().as_dict())
