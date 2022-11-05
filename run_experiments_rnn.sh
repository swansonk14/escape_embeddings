#!/bin/bash

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/24 \
    --split_type mutation \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/25 \
    --split_type site \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/26 \
    --split_type mutation \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type rnn \
    --save_dir results/27 \
    --split_type site \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/28 \
    --split_type mutation \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/29 \
    --split_type site \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/30 \
    --split_type antibody \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/31 \
    --split_type antibody_group \
    --task_type classification \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/32 \
    --split_type mutation \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/33 \
    --split_type site \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/34 \
    --split_type antibody \
    --task_type regression \
    --device cuda \
    --skip_existing

python predict_escape.py \
    --antibody_path data/antibodies.csv \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type rnn \
    --save_dir results/35 \
    --split_type antibody_group \
    --task_type regression \
    --device cuda \
    --skip_existing
