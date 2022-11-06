"""Contains an abstract PyTorch escape model class."""
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from constants import TASK_TYPE_OPTIONS
from dataset import MutationDataset
from models_escape import EscapeModel


class PyTorchEscapeModel(EscapeModel):
    """An abstract PyTorch model that predicts escape."""

    def __init__(self,
                 task_type: TASK_TYPE_OPTIONS,
                 num_epochs: int,
                 batch_size: int,
                 device: str) -> None:
        """Initialize the model.

        :param task_type: The type of task to perform, i.e., classification or regression.
        :param num_epochs: The number of epochs to train for.
        :param batch_size: The number of sequences to process at once.
        :param device: The device to use (e.g., "cpu" or "cuda") for the model.
        """
        super(PyTorchEscapeModel, self).__init__(task_type=task_type)

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        # Ensure PyTorch reproducibility
        torch.manual_seed(0)

        # Create loss function
        if self.task_type == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f'Task type "{self.task_type}" is not supported.')

    @property
    @abstractmethod
    def core_model(self) -> nn.Module:
        """Gets the core PyTorch model."""
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Adam:
        """Gets the optimizer."""
        pass

    @abstractmethod
    def collate_batch(self, input_tuples: list[tuple[str, int, str, Optional[float]]]
                      ) -> tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        """Collate antibody and/or antigen sequences/embeddings and escape scores for input to the model.

        :param input_tuples: A list of tuples of antibody, site, mutation, and escape score (if available).
        :return: A tuple with a Tensor containing the antibody and/or antigen sequences/embeddings
                 and a FloatTensor of escape scores (if available).
        """
        pass

    def fit(self,
            antibodies: list[str],
            sites: list[int],
            wildtypes: list[str],
            mutants: list[str],
            escapes: list[float]) -> 'PyTorchEscapeModel':
        """Fits the model on the training escape data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :param escapes: A list of escape scores for each mutation at each site.
        :return: The fitted model.
        """
        # Create mutation dataset
        dataset = MutationDataset(
            antibodies=antibodies,
            sites=sites,
            mutants=mutants,
            escapes=escapes
        )

        # Create data loader
        generator = torch.Generator()
        generator.manual_seed(0)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
            generator=generator
        )

        # Train
        self.core_model.train()
        for _ in trange(self.num_epochs, desc='Epochs', leave=False):
            for batch_data, batch_escape in tqdm(data_loader, total=len(data_loader), desc='Batches', leave=False):
                self.core_model.zero_grad()

                if isinstance(batch_data, tuple):
                    batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                else:
                    batch_data = batch_data.to(self.device)

                batch_escape = batch_escape.to(self.device)

                preds = self.core_model(batch_data)

                loss = self.loss_fn(preds, batch_escape)

                loss.backward()
                self.optimizer.step()

        return self

    def predict(self,
                antibodies: list[str],
                sites: list[int],
                wildtypes: list[str],
                mutants: list[str]) -> np.ndarray:
        """Makes escape predictions on the test data.

        :param antibodies: A list of antibodies.
        :param sites: A list of mutated sites.
        :param wildtypes: A list of wildtype amino acids at each site.
        :param mutants: A list of mutant amino acids at each site.
        :return: A numpy array of predicted escape scores.
        """
        # Create mutation dataset
        dataset = MutationDataset(
            antibodies=antibodies,
            sites=sites,
            mutants=mutants
        )

        # Create data loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch
        )

        # Predict
        all_preds = []

        self.core_model.eval()
        with torch.no_grad():
            for batch_data, _ in tqdm(data_loader, total=len(data_loader), desc='Batches', leave=False):
                if isinstance(batch_data, tuple):
                    batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                else:
                    batch_data = batch_data.to(self.device)

                preds = self.core_model(batch_data)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        return all_preds

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return str(self.core_model)
