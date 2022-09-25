# Escape Embeddings

Modeling SARS-CoV-2 antigen escape from antibodies using pretrained protein language model embeddings.


## Installation

Install conda environment.
```bash
conda env create -f environment.yml
```

Activate conda environment.
```bash
conda activate escape_embeddings
```


## Data

Download single amino acid mutation data for SARS-CoV-2 receptor binding domain (RBD) from here: https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps/tree/main/data/2022_Cao_Omicron

Source: Cao et al., Omicron escapes the majority of existing SARS-CoV-2 neutralizing antibodies, Nature, 2022. https://www.nature.com/articles/s41586-021-04385-3

```bash
wget -P data https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/main/data/2022_Cao_Omicron/antibodies.csv
wget -P data https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/main/data/2022_Cao_Omicron/data.csv
```


## Extract RBD

Extract the SARS-CoV-2 RBD sequence and site to amino acid mapping.

```bash
python extract_rbd.py \
    --data_path data/data.csv
```


## Visualize Escape

Visualize escape mutation patterns.

```bash
python visualize_escape.py \
    --data_path data/data.csv \
    --antibody_path data/antibodies.csv \
    --save_dir plots/escape
```


## Generate ESM2 Embeddings

Generate antibody and antigen embeddings with an ESM2 model. See ESM2 model options at https://github.com/facebookresearch/esm.

Antigen embeddings
```bash
python generate_embeddings.py \
    --hub_dir models \
    --esm_model esm2_t6_8M_UR50D \
    --embedding_type antigen \
    --save_path embeddings/antigen_name_to_embedding.pt
```

Antibody embeddings
```bash
python generate_embeddings.py \
    --hub_dir models \
    --esm_model esm2_t6_8M_UR50D \
    --embedding_type antibody \
    --save_path embeddings/antibody_name_to_embedding.pt \
    --antibody_path data/antibodies.csv
```

Antigen embeddings
```bash
python generate_embeddings.py \
    --hub_dir models \
    --esm_model esm2_t6_8M_UR50D \
    --embedding_type antibody-antigen \
    --save_path embeddings/antibody_antigen_name_to_embedding.pt \
    --antibody_path data/antibodies.csv
```
