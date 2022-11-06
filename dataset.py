"""Contains a mutation dataset class."""
from typing import Optional

from torch.utils.data import Dataset


class MutationDataset(Dataset):
    """A dataset for mutations containing antibodies, sites, and mutations."""

    def __init__(self,
                 antibodies: list[str],
                 sites: list[int],
                 mutants: list[str],
                 escapes: Optional[list[float]] = None) -> None:
        """Initialize the dataset.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        """
        escapes = escapes if escapes is not None else [None] * len(antibodies)

        assert len(antibodies) == len(sites) == len(mutants) == len(escapes)

        self.antibodies = antibodies
        self.sites = sites
        self.mutants = mutants
        self.escapes = escapes

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.antibodies)

    def __getitem__(self, item: int | slice) -> tuple[str, int, str, Optional[float]] \
                                                | list[tuple[str, int, str, Optional[float]]]:
        """Get an item from the dataset.

        :param item: An int or slice to index into the dataset.
        :return: An item or list of items from the dataset corresponding to the provided index or slice.
                 Each item is a tuple of an antibody, a site, a mutation, and an escape score (if available).
        """
        if isinstance(item, int):
            return self.antibodies[item], self.sites[item], self.mutants[item], self.escapes[item]

        if isinstance(item, slice):
            return list(zip(*[self.antibodies[item], self.sites[item], self.mutants[item], self.escapes[item]]))

        raise NotImplementedError(f'__getitem__ with item type "{type(item)}" is not supported.')
