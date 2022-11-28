#!/bin/bash

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/138 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/139 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/140 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/141 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/142 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/143 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/144 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/145 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/146 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/147 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/148 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/149 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/150 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/151 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/152 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/153 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/154 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/155 \
    --split_type mutation \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/156 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/157 \
    --split_type site \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/158 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/159 \
    --split_type antibody \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/160 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/161 \
    --split_type antibody_group \
    --task_type regression \
    --skip_existing

python predict_escape.py \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/162 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity per-antibody \
    --model_type embedding \
    --save_dir results/163 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/164 \
    --split_type mutation \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/165 \
    --split_type site \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/166 \
    --split_type antibody \
    --task_type classification \
    --skip_existing

python predict_escape.py \
    --antibody_embedding_granularity residue \
    --antibody_embedding_type attention \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --data_path data/data.csv \
    --device cuda \
    --model_granularity cross-antibody \
    --model_type embedding \
    --save_dir results/167 \
    --split_type antibody_group \
    --task_type classification \
    --skip_existing

