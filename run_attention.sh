#!/bin/bash

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/attention_mutation_per_antibody_residue \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/attention_site_per_antibody_residue \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/attention_mutation_cross_antibody_residue \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/attention_site_cross_antibody_residue \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/attention_antibody_cross_antibody_residue \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/attention_antibody_group_cross_antibody_residue \
    --device cuda \
    --skip_existing

