# Escape Embeddings

Modeling SARS-CoV-2 antigen escape from antibodies using pretrained protein language model embeddings.


## Installation

Install conda environment.
```
conda env create -f environment.yml
```

Activate conda environment.
```
conda activate escape_embeddings
```


## Data

Download single amino acid mutation data for SARS-CoV-2 receptor binding domain (RBD) from here: https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps/tree/main/data/2022_Cao_Omicron

Source: Cao et al., Omicron escapes the majority of existing SARS-CoV-2 neutralizing antibodies, Nature, 2022. https://www.nature.com/articles/s41586-021-04385-3

```
wget -P data https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/main/data/2022_Cao_Omicron/antibodies.csv
wget -P data https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/main/data/2022_Cao_Omicron/data.csv
```


## Extract RBD

Extract the SARS-CoV-2 RBD sequence and site to amino acid mapping.

```
python extract_rbd.py \
    --data_path data/data.csv
```


## Visualize Escape

Visualize escape mutation patterns.

```
python visualize_escape.py \
    --data_path data/data.csv \
    --antibodies_path data/antibodies.csv \
    --save_dir plots/escape
```
