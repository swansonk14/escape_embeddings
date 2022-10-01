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
    --antibody_path data/antibodies.csv \
    --save_dir results/23 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/24 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/25 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --save_dir results/26 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --save_dir results/27 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/28 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/29 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --save_dir results/30 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --save_dir results/31 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --save_dir results/32 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --save_dir results/33 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --save_dir results/34 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --save_dir results/35 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/36 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/37 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/38 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/39 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/40 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type likelihood \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/41 \
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
    --save_dir results/42 \
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
    --save_dir results/43 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/44 \
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
    --save_dir results/45 \
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
    --save_dir results/46 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/47 \
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
    --save_dir results/48 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/49 \
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
    --save_dir results/50 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/51 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/52 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/53 \
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
    --save_dir results/54 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/55 \
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
    --save_dir results/56 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/57 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/58 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/59 \
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
    --save_dir results/60 \
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
    --save_dir results/61 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/62 \
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
    --save_dir results/63 \
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
    --save_dir results/64 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/65 \
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
    --save_dir results/66 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/67 \
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
    --save_dir results/68 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/69 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/70 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/71 \
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
    --save_dir results/72 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/73 \
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
    --save_dir results/74 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/75 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/76 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/77 \
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
    --save_dir results/78 \
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
    --save_dir results/79 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/80 \
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
    --save_dir results/81 \
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
    --save_dir results/82 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/83 \
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
    --save_dir results/84 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/85 \
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
    --save_dir results/86 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/87 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/88 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/89 \
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
    --save_dir results/90 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/91 \
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
    --save_dir results/92 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/93 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/94 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/95 \
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
    --save_dir results/96 \
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
    --save_dir results/97 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/98 \
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
    --save_dir results/99 \
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
    --save_dir results/100 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/101 \
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
    --save_dir results/102 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/103 \
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
    --save_dir results/104 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/105 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/106 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/107 \
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
    --save_dir results/108 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/109 \
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
    --save_dir results/110 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/111 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/112 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity per-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/113 \
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
    --save_dir results/114 \
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
    --save_dir results/115 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/116 \
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
    --save_dir results/117 \
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
    --save_dir results/118 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/119 \
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
    --save_dir results/120 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/121 \
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
    --save_dir results/122 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/123 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/124 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/125 \
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
    --save_dir results/126 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/127 \
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
    --save_dir results/128 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/129 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/130 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/131 \
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
    --save_dir results/132 \
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
    --save_dir results/133 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/134 \
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
    --save_dir results/135 \
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
    --save_dir results/136 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/137 \
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
    --save_dir results/138 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/139 \
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
    --save_dir results/140 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/141 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/142 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/143 \
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
    --save_dir results/144 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/145 \
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
    --save_dir results/146 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/147 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/148 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/149 \
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
    --save_dir results/150 \
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
    --save_dir results/151 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/152 \
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
    --save_dir results/153 \
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
    --save_dir results/154 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/155 \
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
    --save_dir results/156 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/157 \
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
    --save_dir results/158 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/159 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/160 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/161 \
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
    --save_dir results/162 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/163 \
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
    --save_dir results/164 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/165 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/166 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/167 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --save_dir results/168 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --save_dir results/169 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/170 \
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
    --save_dir results/171 \
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
    --antigen_embedding_type difference \
    --save_dir results/172 \
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
    --antigen_embedding_type mutant_difference \
    --save_dir results/173 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/174 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/175 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/176 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/177 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/178 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/179 \
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
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/180 \
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
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/181 \
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
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/182 \
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
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/183 \
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
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/184 \
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
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/185 \
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
    --save_dir results/186 \
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
    --save_dir results/187 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/188 \
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
    --save_dir results/189 \
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
    --save_dir results/190 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/191 \
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
    --save_dir results/192 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/193 \
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
    --save_dir results/194 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/195 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/196 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/197 \
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
    --save_dir results/198 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/199 \
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
    --save_dir results/200 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/201 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/202 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/203 \
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
    --save_dir results/204 \
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
    --save_dir results/205 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/206 \
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
    --save_dir results/207 \
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
    --save_dir results/208 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/209 \
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
    --save_dir results/210 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/211 \
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
    --save_dir results/212 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/213 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/214 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/215 \
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
    --save_dir results/216 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/217 \
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
    --save_dir results/218 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/219 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/220 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/221 \
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
    --save_dir results/222 \
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
    --save_dir results/223 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/224 \
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
    --save_dir results/225 \
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
    --save_dir results/226 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/227 \
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
    --save_dir results/228 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/229 \
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
    --save_dir results/230 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/231 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/232 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/233 \
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
    --save_dir results/234 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/235 \
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
    --save_dir results/236 \
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
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/237 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/238 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/239 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --save_dir results/240 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --save_dir results/241 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --save_dir results/242 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --save_dir results/243 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --save_dir results/244 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --save_dir results/245 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/246 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/247 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/248 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/249 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/250 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/251 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/252 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/253 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/254 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/255 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/256 \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type embedding \
    --model_granularity cross-antibody \
    --task_type regression \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/257 \
    --skip_existing

