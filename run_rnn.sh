#!/bin/bash

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/rnn_mutation_per_antibody \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity per-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/rnn_site_per_antibody \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type mutation \
    --save_dir results/rnn_mutation_cross_antibody \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type site \
    --save_dir results/rnn_site_cross_antibody \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody \
    --save_dir results/rnn_antibody_cross_antibody \
    --skip_existing

python predict_escape.py \
    --data_path data/data.csv \
    --model_type rnn \
    --model_granularity cross-antibody \
    --task_type classification \
    --split_type antibody_group \
    --antibody_path data/antibodies.csv \
    --save_dir results/rnn_antibody_group_cross_antibody \
    --skip_existing
