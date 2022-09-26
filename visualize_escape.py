"""Visualize antibody escape in response to SARS-CoV-2 RBD mutations."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from constants import (
    ANTIBODY_COLUMN,
    ANTIBODY_NAME_COLUMN,
    EPITOPE_GROUP_COLUMN,
    ESCAPE_COLUMN,
    MUTANT_COLUMN,
    RBD_SEQUENCE,
    SITE_COLUMN,
    WILDTYPE_COLUMN
)


def visualize_escape_score_histogram(data: pd.DataFrame, save_path: Path) -> None:
    """Visualize escape scores as a histogram.

    :param data: DataFrame containing escape data.
    :param save_path: Path to PDF/PNG file where escape score histogram will be saved.
    """
    plt.hist(data[data[ESCAPE_COLUMN] != 0][ESCAPE_COLUMN], bins=100)
    plt.xlabel('Escape Score')
    plt.ylabel('Count')
    plt.title('Nonzero Escape Scores')

    plt.tight_layout()
    plt.savefig(save_path)


def visualize_escape_by_antibody_site(data: pd.DataFrame, save_path: Path) -> None:
    """Visualize escape per antibody per site.

    :param data: DataFrame containing escape data.
    :param save_path: Path to PDF/PNG file where escape score plot by antibody site will be saved.
    """
    escape = data.groupby([ANTIBODY_COLUMN, SITE_COLUMN])[
        ESCAPE_COLUMN].max().reset_index()  # max escape per antibody per site

    antibodies = escape[ANTIBODY_COLUMN].unique()
    sites = escape[SITE_COLUMN].unique()
    assert len(sites) == len(RBD_SEQUENCE)

    escape_grid = escape[ESCAPE_COLUMN].to_numpy().reshape(len(antibodies), len(sites))

    fig, ax = plt.subplots()
    im = ax.imshow(escape_grid, cmap=plt.get_cmap('viridis'))
    fig.colorbar(im)

    ax.set_xticks(np.arange(len(sites)), RBD_SEQUENCE, fontsize=1)
    ax.set_xlabel('RBD Sequence')
    ax.set_yticks(np.arange(len(antibodies)), antibodies, fontsize=1)
    ax.set_ylabel('Antibody')

    plt.title('Escape Score per Antibody across RBD')
    plt.tight_layout()
    plt.savefig(save_path)


def visualize_escape_by_antibody_site_by_group(data: pd.DataFrame, antibody_data: pd.DataFrame,
                                               save_path: Path) -> None:
    """Visualize escape per antibody per site by antibody group.

    :param data: DataFrame containing escape data.
    :param antibody_data: DataFrame containing antibody data (including antibody groups).
    :param save_path: Path to PDF/PNG file where escape score plot by antibody site by group will be saved.
    """
    escape = data.groupby([ANTIBODY_COLUMN, SITE_COLUMN])[
        ESCAPE_COLUMN].max().reset_index()  # max escape per antibody per site

    antibodies = escape[ANTIBODY_COLUMN].unique()
    sites = escape[SITE_COLUMN].unique()
    assert len(sites) == len(RBD_SEQUENCE)

    escape_grid = escape[ESCAPE_COLUMN].to_numpy().reshape(len(antibodies), len(sites))
    min_escape, max_escape = escape_grid.min(), escape_grid.max()

    escape_data = pd.DataFrame(escape_grid, index=antibodies, columns=sites)

    groups = sorted(antibody_data[EPITOPE_GROUP_COLUMN].unique())
    assert len(groups) == 6

    fig, axes = plt.subplots(3, 2)

    for i, group in enumerate(groups):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        group_antibodies = antibody_data[antibody_data[EPITOPE_GROUP_COLUMN] == group][ANTIBODY_NAME_COLUMN]
        group_escape_grid = escape_data.loc[escape_data.index.isin(group_antibodies)].to_numpy()
        im = ax.imshow(group_escape_grid, cmap=plt.get_cmap('viridis'), vmin=min_escape, vmax=max_escape)

        ax.set_xticks(np.arange(len(sites)), RBD_SEQUENCE, fontsize=1)
        if row == 2:
            ax.set_xlabel('RBD Sequence')
        ax.set_yticks(np.arange(len(group_antibodies)), group_antibodies, fontsize=1)
        if col == 0:
            ax.set_ylabel('Antibody')
        ax.set_title(f'Epitope Group {group}')

    fig.colorbar(im, ax=axes.ravel().tolist())

    fig.suptitle('Escape Score per Antibody across RBD by Epitope Group')
    plt.savefig(save_path, dpi=1200)


def visualize_escape_by_amino_acid_change(data: pd.DataFrame, save_path: Path) -> None:
    """Visualize escape per amino acid change (pair of old and new).

    :param data: DataFrame containing escape data.
    :param save_path: Path to PDF/PNG file where escape score plot by amino acid change will be saved.
    """
    escape = data.groupby([WILDTYPE_COLUMN, MUTANT_COLUMN])[
        ESCAPE_COLUMN].mean().reset_index()  # mean escape per aa change

    wildtype = escape[WILDTYPE_COLUMN].unique()
    mutation = escape[MUTANT_COLUMN].unique()

    escape_grid = escape[ESCAPE_COLUMN].to_numpy().reshape(len(wildtype), len(mutation))

    fig, ax = plt.subplots()
    im = ax.imshow(escape_grid, cmap=plt.get_cmap('viridis'))
    fig.colorbar(im)

    ax.set_xticks(np.arange(len(mutation)), mutation)
    ax.set_xlabel(MUTANT_COLUMN)
    ax.set_yticks(np.arange(len(wildtype)), wildtype)
    ax.set_ylabel(WILDTYPE_COLUMN)

    plt.title('Escape Score by Amino Acid Change')
    plt.tight_layout()
    plt.savefig(save_path)


def visualize_escape(data_path: Path, antibody_path: Path, save_dir: Path) -> None:
    """Visualize antibody escape in response to SARS-CoV-2 RBD mutations.

    :param data_path: Path to CSV file containing mutation data.
    :param antibody_path: Path to CSV file containing antibody data.
    :param save_dir: Path to directory where plots will be saved.
    """
    # Load data
    data = pd.read_csv(data_path)
    antibody_data = pd.read_csv(antibody_path)

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


if __name__ == '__main__':
    class Args(Tap):
        data_path: Path
        """Path to CSV file containing mutation data."""
        antibody_path: Path
        """Path to CSV file containing antibody data."""
        save_dir: Path
        """Path to directory where plots will be saved."""


    visualize_escape(**Args().parse_args().as_dict())
