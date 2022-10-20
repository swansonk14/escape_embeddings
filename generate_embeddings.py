"""Generate antigen/antibody embeddings using the ESM2 model from https://github.com/facebookresearch/esm."""
from time import time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import torch
from esm import Alphabet, BatchConverter, ESM2
from tap import Tap
from tqdm import trange

from constants import (
    AA_ALPHABET,
    ANTIBODY_CHAINS,
    ANTIBODY_NAME_COLUMN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    HEAVY_CHAIN,
    HEAVY_CHAIN_COLUMN,
    LIGHT_CHAIN,
    LIGHT_CHAIN_COLUMN,
    PROTEIN_LINKER,
    RBD_SEQUENCE,
    RBD_START_SITE
)


def get_antigen_sequences() -> list[tuple[str, str]]:
    """Get antigen sequences, one for the wildtype and one for each single point mutation.

    :return: A list of tuples of antigen (name, sequence), where the first is the wildtype
             and the others contain single point mutations. Names are either 'wildtype' or site_mutation.
    """
    antigen_sequences = [('wildtype', RBD_SEQUENCE)]
    for i in range(len(RBD_SEQUENCE)):
        wildtype_aa = RBD_SEQUENCE[i]

        for mutant_aa in sorted(set(AA_ALPHABET) - {wildtype_aa}):
            mutant_sequence = RBD_SEQUENCE[:i] + mutant_aa + RBD_SEQUENCE[i + 1:]
            antigen_sequences.append((f'{RBD_START_SITE + i}_{mutant_aa}', mutant_sequence))

    return antigen_sequences


def get_antibody_sequences(antibody_path: Path) -> dict[str, dict[str, str]]:
    """Get antibody sequences, two for each antibody (heavy and light chain).

    :param antibody_path: Path to a file containing antibody sequences.
    :return: A dictionary mapping antibody name to a dictionary mapping 'heavy' or 'light' to antibody sequence.
    """
    # Load antibody data
    antibody_data = pd.read_csv(antibody_path)

    # Get antibody sequences
    antibody_sequences = {
        name: {
            HEAVY_CHAIN: heavy_sequence.rstrip('*'),
            LIGHT_CHAIN: light_sequence.rstrip('*')
        } for name, heavy_sequence, light_sequence in zip(antibody_data[ANTIBODY_NAME_COLUMN],
                                                          antibody_data[HEAVY_CHAIN_COLUMN],
                                                          antibody_data[LIGHT_CHAIN_COLUMN])
    }

    return antibody_sequences


def load_esm_model(hub_dir: str, esm_model: str) -> tuple[ESM2, Alphabet, BatchConverter]:
    """Load an ESM2 model and batch converter.

    :param hub_dir: Path to directory where torch hub models are saved.
    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :return: A tuple of a pretrained ESM2 model and a BatchConverter for preparing protein sequences as input.
    """
    torch.hub.set_dir(hub_dir)
    model, alphabet = torch.hub.load('facebookresearch/esm:main', esm_model)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    return model, alphabet, batch_converter


def generate_esm_embeddings(
        model: ESM2,
        last_layer: int,
        batch_converter: BatchConverter,
        sequences: list[tuple[str, str]],
        average_embeddings: bool = False,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, torch.FloatTensor]:
    """Generate embeddings using an ESM2 model from https://github.com/facebookresearch/esm.

    :param model: A pretrained ESM2 model.
    :param last_layer: Last layer of the ESM2 model, which will be used to extract embeddings.
    :param batch_converter: A BatchConverter for preparing protein sequences as input.
    :param sequences: A list of tuples of (name, sequence) for the proteins.
    :param average_embeddings: Whether to average the residue embeddings for each protein.
    :param device: The device to use (e.g., "cpu" or "cuda") for the model.
    :param batch_size: The number of sequences to process at once.
    :return: A dictionary mapping protein name to per-residue ESM2 embedding.
    """
    # Move model to device
    model = model.to(device)

    # Compute all embeddings
    start = time()
    name_to_embedding = {}

    with torch.no_grad():
        # Iterate over batches of sequences
        for i in trange(0, len(sequences), batch_size):
            # Get batch of sequences
            batch_sequences = sequences[i:i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)
            batch_tokens = batch_tokens.to(device)

            # Compute embeddings
            results = model(batch_tokens, repr_layers=[last_layer], return_contacts=False)

            # Get per-residue embeddings
            batch_embeddings = results['representations'][last_layer].cpu()

            # Map sequence name to embedding
            for (name, sequence), embedding in zip(batch_sequences, batch_embeddings):
                # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1
                embedding = embedding[1:len(sequence) + 1]

                # Optionally average embeddings for each sequence
                if average_embeddings:
                    embedding = embedding.mean(dim=0)

                name_to_embedding[name] = embedding

    print(f'Time = {time() - start} seconds for {len(sequences):,} sequences')

    assert len(sequences) == len(name_to_embedding)

    return name_to_embedding


def generate_embeddings(
        hub_dir: str,
        esm_model: str,
        last_layer: int,
        embedding_type: Literal['antigen', 'antibody', 'antibody-antigen'],
        save_path: Path,
        average_embeddings: bool = False,
        antibody_path: Optional[Path] = None,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """Generate antigen/antibody embeddings using the ESM2 model from https://github.com/facebookresearch/esm.

    :param hub_dir: Path to directory where torch hub models are saved.
    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :param last_layer: Last layer of the ESM2 model, which will be used to extract embeddings.
    :param embedding_type: Type of embedding to compute. When using antibody embeddings, must provide antibody_path.
    :param average_embeddings: Whether to average the residue embeddings for each protein.
    :param save_path: Path to PT file where a dictionary mapping protein name to embeddings will be saved.
    :param antibody_path: Path to a file containing antibody sequences.
    :param device: The device to use (e.g., "cpu" or "cuda") for the model.
    :param batch_size: The number of sequences to process at once.
    """
    # Validate parameters
    if 'antibody' in embedding_type and antibody_path is None:
        raise ValueError('Must provide antibody_path if using embedding type that includes antibodies.')

    if 'antibody' not in embedding_type and antibody_path is not None:
        raise ValueError('Do not provide an antibody path if not using an embedding type that includes antibodies.')

    # Get antigen sequences, one for each point mutation plus the wildtype
    antigen_sequences = get_antigen_sequences() if 'antigen' in embedding_type else None

    # Get antibody sequences, two for each antibody (heavy and light chain)
    antibody_sequences = get_antibody_sequences(antibody_path=antibody_path) if 'antibody' in embedding_type else None

    # Get sequences for input to the model
    if embedding_type == 'antigen':
        sequences = antigen_sequences
    elif embedding_type == 'antibody':
        sequences = [
            (f'{antibody_name}_{antibody_chain}',
             antibody_chain_to_sequence[antibody_chain])
            for antibody_name, antibody_chain_to_sequence in antibody_sequences.items()
            for antibody_chain in ANTIBODY_CHAINS
        ]
    elif embedding_type == 'antibody-antigen':
        # Combine antibody and antigen with a protein linker
        sequences = [
            (f'{antibody_name}_{antibody_chain}_{antigen_name}',
             f'{antibody_chain_to_sequence[antibody_chain]}{PROTEIN_LINKER}{antigen_sequence}')
            for antibody_name, antibody_chain_to_sequence in antibody_sequences.items()
            for antibody_chain in ANTIBODY_CHAINS
            for (antigen_name, antigen_sequence) in antigen_sequences
        ]
    else:
        raise ValueError(f'Embedding type "{embedding_type}" is not supported.')

    # Print stats
    print(f'Number of {embedding_type} sequences = {len(sequences):,}')

    # Load ESM-2 model
    model, alphabet, batch_converter = load_esm_model(hub_dir=hub_dir, esm_model=esm_model)

    # Generate embeddings
    sequence_representations = generate_esm_embeddings(
        model=model,
        last_layer=last_layer,
        batch_converter=batch_converter,
        sequences=sequences,
        average_embeddings=average_embeddings,
        device=device,
        batch_size=batch_size
    )

    # Save embeddings
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequence_representations, save_path)


if __name__ == '__main__':
    class Args(Tap):
        hub_dir: str
        """Path to directory where torch hub models are saved."""
        esm_model: str
        """Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm."""
        last_layer: int
        """Last layer of the ESM2 model, which will be used to extract embeddings."""
        embedding_type: Literal['antigen', 'antibody', 'antibody-antigen']
        """Type of embedding to compute. When using antibody embeddings, must provide antibody_path."""
        save_path: Path
        """Path to PT file where a dictionary mapping protein name to embeddings will be saved."""
        average_embeddings: bool = False
        """Whether to average the residue embeddings for each protein."""
        antibody_path: Optional[Path] = None
        """Path to a file containing antibody sequences."""
        device: str = DEFAULT_DEVICE
        """The device to use (e.g., "cpu" or "cuda") for the model."""
        batch_size: int = DEFAULT_BATCH_SIZE
        """The number of sequences to process at once."""

    generate_embeddings(**Args().parse_args().as_dict())
