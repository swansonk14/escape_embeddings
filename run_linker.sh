#!/bin/bash

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --save_dir results/linker_mutation_per_antibody_sequence \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --save_dir results/linker_site_per_antibody_sequence \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --save_dir results/linker_mutation_cross_antibody_sequence \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --save_dir results/linker_site_cross_antibody_sequence \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --save_dir results/linker_antibody_cross_antibody_sequence \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --save_dir results/linker_antibody_group_cross_antibody_sequence \
    --skip_existing
