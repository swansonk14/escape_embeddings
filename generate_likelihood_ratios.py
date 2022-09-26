"""Generate likelihood ratios of mutant vs wildtype antigens using the ESM2 model from https://github.com/facebookresearch/esm.
Uses the masked marginals scoring method from https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full.
Code adapted from https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py.
"""
from pathlib import Path

import torch
from tap import Tap
from tqdm import tqdm

from constants import (
    AA_ALPHABET_SET,
    RBD_SEQUENCE,
    RBD_START_SITE
)
from generate_embeddings import load_esm_model


def generate_likelihood_ratios(hub_dir: str, esm_model: str, save_path: Path) -> None:
    """Generate likelihood ratios of mutant vs wildtype antigens using the ESM2 model from https://github.com/facebookresearch/esm.

    :param hub_dir: Path to directory where torch hub models are saved.
    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :param save_path: Path to PT file where a dictionary mapping antigen name to likelihood will be saved.
    """
    # Load ESM-2 model
    model, alphabet, batch_converter = load_esm_model(hub_dir=hub_dir, esm_model=esm_model)

    # Create batch from RBD sequence
    batch_labels, batch_strs, batch_tokens = batch_converter([('RBD', RBD_SEQUENCE)])

    # Generate likelihood ratios
    all_token_probs = []
    for i in tqdm(range(batch_tokens.size(1))):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx

        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked)['logits'], dim=-1)

        all_token_probs.append(token_probs[:, i])  # vocab size

    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

    name_to_likelihood_ratio = {
        f'{RBD_START_SITE + i}_{mutant_aa}': (token_probs[0, 1 + i, alphabet.get_idx(mutant_aa)]
                                              - token_probs[0, 1 + i, alphabet.get_idx(wildtype_aa)]).item()
        for i, wildtype_aa in enumerate(RBD_SEQUENCE)
        for mutant_aa in sorted(AA_ALPHABET_SET - {wildtype_aa})
    }

    # Save embeddings
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(name_to_likelihood_ratio, save_path)


if __name__ == '__main__':
    class Args(Tap):
        hub_dir: str
        """Path to directory where torch hub models are saved."""
        esm_model: str
        """Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm."""
        save_path: Path
        """Path to PT file where a dictionary mapping antigen name to likelihood will be saved."""


    generate_likelihood_ratios(**Args().parse_args().as_dict())
