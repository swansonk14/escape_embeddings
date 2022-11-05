#!/bin/bash

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --save_dir results/0 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --save_dir results/1 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --save_dir results/2 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --save_dir results/3 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/4 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/5 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/6 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/7 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/8 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/9 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/10 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --save_dir results/11 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --save_dir results/12 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --save_dir results/13 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --save_dir results/14 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --save_dir results/15 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/16 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/17 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/18 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/19 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/20 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/21 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/22 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --save_dir results/23 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

# RNN experiments in run_experiments_rnn.sh since they require GPU
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity per-antibody \
#    --model_type rnn \
#    --save_dir results/24 \
#    --split_type mutation \
#    --task_type classification \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity per-antibody \
#    --model_type rnn \
#    --save_dir results/25 \
#    --split_type site \
#    --task_type classification \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity per-antibody \
#    --model_type rnn \
#    --save_dir results/26 \
#    --split_type mutation \
#    --task_type regression \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity per-antibody \
#    --model_type rnn \
#    --save_dir results/27 \
#    --split_type site \
#    --task_type regression \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/28 \
#    --split_type mutation \
#    --task_type classification \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/29 \
#    --split_type site \
#    --task_type classification \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/30 \
#    --split_type antibody \
#    --task_type classification \
#    --skip_existing
#
#python predict_escape.py \
#    --antibody_path data/antibodies.csv \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/31 \
#    --split_type antibody_group \
#    --task_type classification \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/32 \
#    --split_type mutation \
#    --task_type regression \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/33 \
#    --split_type site \
#    --task_type regression \
#    --skip_existing
#
#python predict_escape.py \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/34 \
#    --split_type antibody \
#    --task_type regression \
#    --skip_existing
#
#python predict_escape.py \
#    --antibody_path data/antibodies.csv \
#    --data_path data/data.csv \
#    --model_granularity cross-antibody \
#    --model_type rnn \
#    --save_dir results/35 \
#    --split_type antibody_group \
#    --task_type regression \
#    --skip_existing

python predict_escape.py \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type likelihood \
    --save_dir results/36 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type likelihood \
    --save_dir results/37 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --save_dir results/38 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --save_dir results/39 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --save_dir results/40 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --save_dir results/41 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/42 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/43 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/44 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/45 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/46 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/47 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/48 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/49 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/50 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/51 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/52 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/53 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/54 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/55 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/56 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/57 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/58 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/59 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/60 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/61 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/62 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/63 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/64 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/65 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/66 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/67 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/68 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/69 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/70 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/71 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/72 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/73 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/74 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/75 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/76 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/77 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/78 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/79 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/80 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/81 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/82 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/83 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/84 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/85 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/86 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/87 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/88 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/89 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/90 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/91 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/92 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/93 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/94 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/95 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/96 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/97 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/98 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/99 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/100 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/101 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/102 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/103 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/104 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/105 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/106 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/107 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/108 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/109 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/110 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/111 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/112 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/113 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/114 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/115 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/116 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/117 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/118 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/119 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/120 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/121 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type linker \
    --antigen_embeddings_path embeddings/antibody_antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/122 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/123 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/124 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant_difference \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/125 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

