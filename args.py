"""Contains argument classes for command line argument parsing."""
from pathlib import Path
from typing import Optional

from tap import Tap

from constants import (
    ANTIBODY_EMBEDDING_TYPE_OPTIONS,
    ANTIGEN_EMBEDDING_TYPE_OPTIONS,
    DEFAULT_ATTENTION_NUM_HEADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_HIDDEN_LAYER_DIMS,
    DEFAULT_RNN_HIDDEN_DIM,
    EMBEDDING_GRANULARITY_OPTIONS,
    MODEL_GRANULARITY_OPTIONS,
    MODEL_TYPE_OPTIONS,
    SPLIT_TYPE_OPTIONS,
    TASK_TYPE_OPTIONS,
)


class PredictEscapeArgs(Tap):
    data_path: Path
    """Path to CSV file containing antibody escape data."""
    save_dir: Path
    """Path to directory where results will be saved."""
    model_granularity: MODEL_GRANULARITY_OPTIONS
    """The granularity of the model, either one model per antibody or one model across all antibodies."""
    model_type: MODEL_TYPE_OPTIONS
    """The type of model to train."""
    task_type: TASK_TYPE_OPTIONS
    """The type of task to perform."""
    split_type: SPLIT_TYPE_OPTIONS
    """The type of data split."""
    antibody_path: Optional[Path] = None
    """Path to a CSV file containing antibody sequences and groups."""
    antigen_likelihoods_path: Optional[Path] = None
    """Path to PT file containing a dictionary mapping from antigen name to (mutant - wildtype) likelihood."""
    antigen_embeddings_path: Optional[Path] = None
    """Path to PT file containing a dictionary mapping from antigen name to ESM2 embedding."""
    antigen_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None
    """The granularity of the antigen embeddings, either a sequence average or per-residue embeddings."""
    antigen_embedding_type: Optional[ANTIGEN_EMBEDDING_TYPE_OPTIONS] = None
    """The type of antigen embedding."""
    antibody_embeddings_path: Optional[Path] = None
    """Path to PT file containing a dictionary mapping from antibody name_chain to ESM2 embedding."""
    antibody_embedding_granularity: Optional[EMBEDDING_GRANULARITY_OPTIONS] = None
    """The granularity of the antibody embeddings, either a sequence average or per-residue embeddings."""
    antibody_embedding_type: Optional[ANTIBODY_EMBEDDING_TYPE_OPTIONS] = None
    """Method of including the antibody embeddings with antigen embeddings."""
    hidden_layer_dims: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_DIMS
    """The sizes of the hidden layers of the MLP model that will be trained."""
    rnn_hidden_dim: int = DEFAULT_RNN_HIDDEN_DIM
    """The dimensionality of the RNN model."""
    attention_num_heads: int = DEFAULT_ATTENTION_NUM_HEADS
    """The number of attention heads for the attention antibody embedding type."""
    num_epochs: Optional[int] = None
    """The number of epochs for the embedding model. If None, num_epochs is set based on model_granularity."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """The batch size for the embedding model."""
    device: str = DEFAULT_DEVICE
    """The device to use (e.g., "cpu" or "cuda") for the RNN and embedding models."""
    verbose: bool = False
    """Whether to print additional debug information."""
    skip_existing: bool = False
    """Whether to skip running the code if the save_dir already exists."""
