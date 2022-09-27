#!/bin/bash

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/0 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/1 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --save_dir results/2 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --save_dir results/3 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/4 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/5 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --save_dir results/6 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/7 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --save_dir results/8 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --save_dir results/9 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --save_dir results/10 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type mutation \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/11 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/12 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/13 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --save_dir results/14 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --save_dir results/15 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/16 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/17 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --save_dir results/18 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/19 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --save_dir results/20 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --save_dir results/21 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --save_dir results/22 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type site \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/23 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/24 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/25 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/26 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/27 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/28 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/29 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/30 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/31 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/32 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/33 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/34 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/35 \
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
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/36 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/37 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/38 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/39 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/40 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/41 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/42 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/43 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/44 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/45 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/46 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/47 \
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
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/48 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/49 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/50 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/51 \
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
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/52 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/53 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/54 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/55 \
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
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/56 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/57 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/58 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/59 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/60 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/61 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/62 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/63 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/64 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/65 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/66 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/67 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/68 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/69 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/70 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/71 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/72 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/73 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/74 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/75 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/76 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/77 \
    --skip_existing

