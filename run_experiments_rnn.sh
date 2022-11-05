#!/bin/bash

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/24 \
    --split_type mutation \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/25 \
    --split_type mutation \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/26 \
    --split_type site \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/27 \
    --split_type site \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/28 \
    --split_type mutation \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/29 \
    --split_type mutation \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/30 \
    --split_type site \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/31 \
    --split_type site \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/32 \
    --split_type mutation \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/33 \
    --split_type mutation \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/34 \
    --split_type site \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/35 \
    --split_type site \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/36 \
    --split_type antibody \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/37 \
    --split_type antibody \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/38 \
    --split_type antibody_group \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/39 \
    --split_type antibody_group \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/40 \
    --split_type mutation \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/41 \
    --split_type mutation \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/42 \
    --split_type site \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/43 \
    --split_type site \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/44 \
    --split_type antibody \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/45 \
    --split_type antibody \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity sequence \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/46 \
    --split_type antibody_group \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --antigen_embedding_granularity residue \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/47 \
    --split_type antibody_group \
    --task_type regression \
    --device cuda \
    --skip_existing
