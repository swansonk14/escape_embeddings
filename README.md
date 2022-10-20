# Escape Embeddings

Modeling SARS-CoV-2 antigen escape from antibodies using pretrained protein language model embeddings.


## Installation

Install conda environment. If using GPU, first open `environment.yml` and uncomment the line with `cudatoolkit=11.3`.
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

Extract the SARS-CoV-2 RBD sequence and site to amino acid mapping. This has already been done and is available in `constants.py`.

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


## ESM2 Protein Language Model

Use an ESM2 protein language model to embed antigens and antibodies and to compute mutant likelihood ratios. See ESM2 model options at https://github.com/facebookresearch/esm.


### Generate ESM2 Embeddings

Generate antibody and antigen embeddings with an ESM2 model.

Antigen embeddings
```bash
python generate_embeddings.py \
    --hub_dir models \
    --esm_model esm2_t33_650M_UR50D \
    --last_layer 33 \
    --embedding_type antigen \
    --save_path embeddings/antigen_embeddings.pt
```

Antibody embeddings
```bash
python generate_embeddings.py \
    --hub_dir models \esm2_t33_650M_UR50D
    --esm_model esm2_t33_650M_UR50D \
    --last_layer 33 \
    --embedding_type antibody \
    --save_path embeddings/antibody_embeddings.pt \
    --antibody_path data/antibodies.csv
```

Antibody-antigen embeddings
```bash
python generate_embeddings.py \
    --hub_dir models \
    --esm_model esm2_t33_650M_UR50D \
    --last_layer 33 \
    --embedding_type antibody-antigen \
    --save_path embeddings/antibody_antigen_embeddings.pt \
    --antibody_path data/antibodies.csv \
    --average_embeddings
```


### Generate ESM2 Likelihood Ratios

Generate mutant vs wildtype antigen likelihood ratios with an ESM2 model.

```bash
python generate_likelihood_ratios.py \
    --hub_dir models \
    --esm_model esm2_t33_650M_UR50D \
    --save_path embeddings/antigen_likelihood_ratios.pt
```


## Set Up Experiments

Set up a bash script with all the experiments to run.

```bash
python setup_experiments.py \
    --data_path data/data.csv \
    --antibody_path data/antibodies.csv \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --experiment_save_dir results \
    --bash_save_path run_experiments.sh \
    --skip_existing
```

## Run Experiments

```bash
bash run_experiments.sh
```


## Combine Results

```bash
python combine_results.py \
    --results_dir results \
    --save_path results/all_results.csv
```


## Plot Results

```bash
python plot_results.py \
    --results_path results/all_results.csv \
    --save_dir plots/results
```
