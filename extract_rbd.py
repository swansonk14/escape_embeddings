"""Extract the receptor binding domain (RBD) of SARS-CoV-2."""
from pathlib import Path

import pandas as pd
from tap import Tap

from constants import (
    SITE_COLUMN,
    WILDTYPE_COLUMN
)


def extract_rbd(data_path: Path) -> None:
    """Extract the receptor binding domain (RBD) of SARS-CoV-2.

    :param data_path: Path to CSV file containing mutation data.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Extract RBD
    site_to_aa = {}
    for site, aa in zip(data[SITE_COLUMN], data[WILDTYPE_COLUMN]):
        if site in site_to_aa:
            assert site_to_aa[site] == aa
        else:
            site_to_aa[site] = aa

    # Check continuity of RBD
    sites = sorted(site_to_aa)
    start, stop = min(sites), max(sites)
    if sites != list(range(start, stop + 1)):
        raise ValueError('RBD is not contiguous.')

    # Join RBD
    rbd_sequence = ''.join(site_to_aa[site] for site in sites)

    # Print RBD
    print(rbd_sequence)
    print(f'Length = {len(rbd_sequence)}')
    print(f'Range = {start}-{stop}')
    print(site_to_aa)


if __name__ == '__main__':
    class Args(Tap):
        data_path: Path
        """Path to CSV file containing mutation data."""

    extract_rbd(**Args().parse_args().as_dict())
