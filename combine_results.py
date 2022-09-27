"""Combine the results of multiple experiments into a single CSV file."""
import json
from pathlib import Path

import pandas as pd
from tap import Tap

from args import PredictEscapeArgs


def analyze_results(results_dir: Path, save_path: Path) -> None:
    """Analyze the results of multiple experiments."""
    # Get all results
    results_dicts = []
    results_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    for experiment_results_dir in sorted(results_dir.iterdir()):
        # Get args from experiment
        args = PredictEscapeArgs()
        args.load(experiment_results_dir / 'args.json')

        # Get results from experiment
        if args.model_granularity == 'per-antibody':
            with open(experiment_results_dir / 'summary_results.json') as f:
                summary_results = json.load(f)

            results = {
                metric: values[metric]['mean']
                for metric, values in summary_results.items()
            }
        elif args.model_granularity == 'cross-antibody':
            with open(experiment_results_dir / 'results.json') as f:
                results = json.load(f)
        else:
            raise ValueError(f'Model granularity "{args.model_granularity}" is not supported.')

        # Combine experiment results with args
        results_dicts.append({
            'model_granularity': args.model_granularity,
            'model_type': args.model_type,
            'task_type': args.task_type,
            'split_type': args.split_type,
            'antibody_group_method': args.antibody_group_method,
            'antigen_embedding_granularity': args.antibody_embedding_granularity,
            'antigen_embedding_type': args.antibody_embedding_type,
            'antibody_embedding_granularity': args.antibody_embedding_granularity,
            'antibody_embedding_type': args.antibody_embedding_type,
            **results
        })

    # Combine all results
    all_results = pd.DataFrame(results_dicts)

    # Save results
    save_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(save_path, index=False)


if __name__ == '__main__':
    class Args(Tap):
        results_dir: Path  # Path to directory containing results.
        save_path: Path  # Path to CSV file where combined results will be saved.

    analyze_results(**Args().parse_args().as_dict())
