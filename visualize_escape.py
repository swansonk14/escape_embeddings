"""Visualize antibody escape in response to SARS-CoV-2 RBD mutations."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from constants import (
    ANTIBODY_COLUMN,
    ANTIBODY_CONDITION_TO_NAME,
    ANTIBODY_NAME_COLUMN,
    EPITOPE_GROUP_COLUMN,
    ESCAPE_COLUMN,
    MUTANT_COLUMN,
    RBD_SEQUENCE,
    SITE_COLUMN,
    WILDTYPE_COLUMN
)


CBAR_FONTSIZE = 15
DPI = 1200


def visualize_escape_score_histogram(data: pd.DataFrame, save_path: Path) -> None:
    """Visualize escape scores as a histogram.

    :param data: DataFrame containing escape data.
    :param save_path: Path to PDF/PNG file where escape score histogram will be saved.
    """
    plt.hist(data[data[ESCAPE_COLUMN] != 0][ESCAPE_COLUMN], bins=100)
    plt.xlabel('Escape Score', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title('Nonzero Escape Scores', fontsize=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=DPI)


def visualize_escape_by_antibody_site(data: pd.DataFrame, save_path: Path) -> None:
    """Visualize escape per antibody per site.

    :param data: DataFrame containing escape data.
    :param save_path: Path to PDF/PNG file where escape score plot by antibody site will be saved.
    """
    # Get the maximum escape score per antibody per site
    escape = data.groupby([ANTIBODY_COLUMN, SITE_COLUMN])[ESCAPE_COLUMN].max().reset_index()

    # Get the unique antibodies and sites
    antibodies = escape[ANTIBODY_COLUMN].unique()
    sites = escape[SITE_COLUMN].unique()
    assert len(sites) == len(RBD_SEQUENCE)

    # Create a grid of escape scores of antibodies by antigen sites
    escape_grid = escape[ESCAPE_COLUMN].to_numpy().reshape(len(antibodies), len(sites))

    # Plot the escape score grid
    fig, ax = plt.subplots()
    im = ax.imshow(escape_grid, cmap=plt.get_cmap('viridis'), interpolation='none')
    cbar = fig.colorbar(im)
    cbar.set_label('Escape Score', rotation=270, fontsize=CBAR_FONTSIZE, labelpad=25)

    ax.set_rasterized(True)
    ax.set_xticks(np.arange(len(sites)), RBD_SEQUENCE, fontsize=1)
    ax.set_xlabel('RBD Sequence', fontsize=15)
    ax.set_yticks(np.arange(len(antibodies)), antibodies, fontsize=1)
    ax.set_ylabel('Antibody', fontsize=15)

    plt.title('Escape Score per Antibody across RBD', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=DPI)


def visualize_escape_by_antibody_site_by_group(
        data: pd.DataFrame,
        antibody_data: pd.DataFrame,
        save_path: Path
) -> None:
    """Visualize escape per antibody per site by antibody group.

    :param data: DataFrame containing escape data.
    :param antibody_data: DataFrame containing antibody data (including antibody groups).
    :param save_path: Path to PDF/PNG file where escape score plot by antibody site by group will be saved.
    """
    # Get the maximum escape score per antibody per site
    escape = data.groupby([ANTIBODY_COLUMN, SITE_COLUMN])[ESCAPE_COLUMN].max().reset_index()

    # Get the unique antibodies and sites
    antibodies = escape[ANTIBODY_COLUMN].unique()
    sites = escape[SITE_COLUMN].unique()
    assert len(sites) == len(RBD_SEQUENCE)

    # Create a grid of escape scores of antibodies by antigen sites
    escape_grid = escape[ESCAPE_COLUMN].to_numpy().reshape(len(antibodies), len(sites))
    min_escape, max_escape = escape_grid.min(), escape_grid.max()

    escape_data = pd.DataFrame(escape_grid, index=antibodies, columns=sites)

    # Group escape data by antibody group
    groups = sorted(antibody_data[EPITOPE_GROUP_COLUMN].unique())
    assert len(groups) == 6

    # Plot the escape score grids by antibody group
    fig, axes = plt.subplots(3, 2)

    # Plot the escape score grid for each antibody group
    for i, group in enumerate(groups):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        group_antibodies = antibody_data[antibody_data[EPITOPE_GROUP_COLUMN] == group][ANTIBODY_NAME_COLUMN]
        group_escape_grid = escape_data.loc[escape_data.index.isin(group_antibodies)].to_numpy()
        im = ax.imshow(group_escape_grid, cmap=plt.get_cmap('viridis'), interpolation='none',
                       vmin=min_escape, vmax=max_escape)
        ax.set_rasterized(True)

        ax.set_xticks(np.arange(len(sites)), RBD_SEQUENCE, fontsize=1)
        if row == 2:
            ax.set_xlabel('RBD Sequence', fontsize=10)
        ax.set_yticks(np.arange(len(group_antibodies)), group_antibodies, fontsize=1)
        if col == 0:
            ax.set_ylabel('Antibody', fontsize=10)
        ax.set_title(f'Epitope Group {group}', fontsize=10)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label('Escape Score', rotation=270, fontsize=CBAR_FONTSIZE, labelpad=25)

    fig.suptitle('Escape Score per Antibody across RBD by Epitope Group', fontsize=15)
    plt.savefig(save_path, bbox_inches='tight', dpi=DPI)


def visualize_escape_by_amino_acid_change(data: pd.DataFrame, save_path: Path) -> None:
    """Visualize escape per amino acid change (pair of old and new).

    :param data: DataFrame containing escape data.
    :param save_path: Path to PDF/PNG file where escape score plot by amino acid change will be saved.
    """
    # Get the average escape score per amino acid change
    escape = data.groupby([WILDTYPE_COLUMN, MUTANT_COLUMN])[ESCAPE_COLUMN].mean().reset_index()

    # Get the unique wildtype and mutant amino acids
    wildtype = escape[WILDTYPE_COLUMN].unique()
    mutation = escape[MUTANT_COLUMN].unique()

    # Create a grid of escape scores of wildtype amino acids by mutant amino acids
    escape_grid = escape[ESCAPE_COLUMN].to_numpy().reshape(len(wildtype), len(mutation))

    # Plot the escape score grid
    fig, ax = plt.subplots()
    im = ax.imshow(escape_grid, cmap=plt.get_cmap('viridis'), interpolation='none')
    cbar = fig.colorbar(im)
    cbar.set_label('Escape Score', rotation=270, fontsize=CBAR_FONTSIZE, labelpad=25)

    ax.set_rasterized(True)
    ax.set_xticks(np.arange(len(mutation)), mutation)
    ax.set_xlabel('Mutant', fontsize=15)
    ax.set_yticks(np.arange(len(wildtype)), wildtype)
    ax.set_ylabel('Wildtype', fontsize=15)

    plt.title('Mutation Model', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=DPI)


def visualize_escape_by_site(data: pd.DataFrame, save_path: Path) -> None:
    """Visualize escape per antigen site.

    :param data: DataFrame containing escape data.
    :param save_path: Path to PDF/PNG file where escape score plot by antigen site will be saved.
    """
    # Get the average escape per antigen site
    escape = data.groupby(SITE_COLUMN)[ESCAPE_COLUMN].mean().reset_index()

    # Get the unique antigen sites
    sites = escape[SITE_COLUMN].unique()

    # Create a grid of escape scores by antigen site
    escape_grid = escape[ESCAPE_COLUMN].to_numpy().reshape(1, len(sites)).repeat(200, axis=0)

    # Plot the escape score by antigen site
    fig, ax = plt.subplots()
    im = ax.imshow(escape_grid, cmap=plt.get_cmap('viridis'), interpolation='none')
    cbar = fig.colorbar(im)
    cbar.set_label('Escape Score', rotation=270, fontsize=CBAR_FONTSIZE, labelpad=25)

    # Place x ticks at every 10th site
    ax.set_rasterized(True)
    ax.set_xticks(np.arange(0, len(sites), 10), sites[::10])
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('Site', fontsize=15)
    ax.set_yticks([])

    plt.title('Site Model', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=DPI)


def visualize_escape(data_path: Path, antibody_path: Path, save_dir: Path) -> None:
    """Visualize antibody escape in response to SARS-CoV-2 RBD mutations.

    :param data_path: Path to CSV file containing mutation data.
    :param antibody_path: Path to CSV file containing antibody data.
    :param save_dir: Path to directory where plots will be saved.
    """
    # Load data
    data = pd.read_csv(data_path)
    antibody_data = pd.read_csv(antibody_path)

    # Correct antibody names to match antibody sequence data and embeddings
    data[ANTIBODY_COLUMN] = [ANTIBODY_CONDITION_TO_NAME.get(antibody, antibody) for antibody in data[ANTIBODY_COLUMN]]

    # Handle outliers
    data.loc[data[ESCAPE_COLUMN] > 1, ESCAPE_COLUMN] = 1.0

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Visualize escape
    visualize_escape_score_histogram(
        data=data,
        save_path=save_dir / 'escape_score_histogram.pdf'
    )
    visualize_escape_by_antibody_site(
        data=data,
        save_path=save_dir / 'escape_by_antibody_site.pdf'
    )
    visualize_escape_by_antibody_site_by_group(
        data=data,
        antibody_data=antibody_data,
        save_path=save_dir / 'escape_by_antibody_site_by_group.pdf'
    )
    visualize_escape_by_amino_acid_change(
        data=data,
        save_path=save_dir / 'escape_by_amino_acid_change.pdf'
    )
    visualize_escape_by_site(
        data=data,
        save_path=save_dir / 'escape_by_antigen_site.pdf'
    )


if __name__ == '__main__':
    class Args(Tap):
        data_path: Path
        """Path to CSV file containing mutation data."""
        antibody_path: Path
        """Path to CSV file containing antibody data."""
        save_dir: Path
        """Path to directory where plots will be saved."""


    visualize_escape(**Args().parse_args().as_dict())
