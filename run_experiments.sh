#!/bin/bash

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --task_type classification \
    --split_type mutation \
    --save_dir results/model_granularity=per-antibody,model_type=mutation,task_type=classification,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --task_type classification \
    --split_type site \
    --save_dir results/model_granularity=per-antibody,model_type=mutation,task_type=classification,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --task_type regression \
    --split_type mutation \
    --save_dir results/model_granularity=per-antibody,model_type=mutation,task_type=regression,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type mutation \
    --task_type regression \
    --split_type site \
    --save_dir results/model_granularity=per-antibody,model_type=mutation,task_type=regression,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --task_type classification \
    --split_type mutation \
    --save_dir results/model_granularity=per-antibody,model_type=site,task_type=classification,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --task_type classification \
    --split_type site \
    --save_dir results/model_granularity=per-antibody,model_type=site,task_type=classification,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --task_type regression \
    --split_type mutation \
    --save_dir results/model_granularity=per-antibody,model_type=site,task_type=regression,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type site \
    --task_type regression \
    --split_type site \
    --save_dir results/model_granularity=per-antibody,model_type=site,task_type=regression,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type likelihood \
    --task_type regression \
    --split_type mutation \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/model_granularity=per-antibody,model_type=likelihood,task_type=regression,split_type=mutation,antigen_likelihoods_path=embeddings/antigen_likelihood_ratios.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type likelihood \
    --task_type regression \
    --split_type site \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/model_granularity=per-antibody,model_type=likelihood,task_type=regression,split_type=site,antigen_likelihoods_path=embeddings/antigen_likelihood_ratios.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity per-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=per-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type classification \
    --split_type mutation \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=classification,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type classification \
    --split_type site \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=classification,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type classification \
    --split_type antibody \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=classification,split_type=antibody 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=classification,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type regression \
    --split_type mutation \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=regression,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type regression \
    --split_type site \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=regression,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type regression \
    --split_type antibody \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=regression,split_type=antibody 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type mutation \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/model_granularity=cross-antibody,model_type=mutation,task_type=regression,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type classification \
    --split_type mutation \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=classification,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type classification \
    --split_type site \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=classification,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type classification \
    --split_type antibody \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=classification,split_type=antibody 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type classification \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=classification,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type regression \
    --split_type mutation \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=regression,split_type=mutation 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type regression \
    --split_type site \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=regression,split_type=site 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type regression \
    --split_type antibody \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=regression,split_type=antibody 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type site \
    --task_type regression \
    --split_type antibody_group \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/model_granularity=cross-antibody,model_type=site,task_type=regression,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --task_type regression \
    --split_type mutation \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=likelihood,task_type=regression,split_type=mutation,antigen_likelihoods_path=embeddings/antigen_likelihood_ratios.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --task_type regression \
    --split_type site \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=likelihood,task_type=regression,split_type=site,antigen_likelihoods_path=embeddings/antigen_likelihood_ratios.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --task_type regression \
    --split_type antibody \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=likelihood,task_type=regression,split_type=antibody,antigen_likelihoods_path=embeddings/antigen_likelihood_ratios.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type likelihood \
    --task_type regression \
    --split_type antibody_group \
    --antigen_likelihoods_path embeddings/antigen_likelihood_ratios.pt \
    --antibody_group_method escape \
    --antibody_path data/antibodies.csv \
    --save_dir results/model_granularity=cross-antibody,model_type=likelihood,task_type=regression,split_type=antibody_group,antigen_likelihoods_path=embeddings/antigen_likelihood_ratios.pt,antibody_group_method=escape,antibody_path=data/antibodies.csv 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type classification \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=classification,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type mutation \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=mutation,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type site \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=site,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity sequence \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type mutant \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
    --task_type regression \
    --split_type antibody \
    --antigen_embeddings_path embeddings/antigen_embeddings.pt \
    --antigen_embedding_granularity residue \
    --antigen_embedding_type difference \
    --antibody_embedding_granularity sequence \
    --antibody_embedding_type concatenation \
    --antibody_embeddings_path embeddings/antibody_embeddings.pt \
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=sequence,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=mutant,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

python predict_escape.py \
    --data_path data/data.csv \
    --model_granularity cross-antibody \
    --model_type embedding \
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
    --save_dir results/model_granularity=cross-antibody,model_type=embedding,task_type=regression,split_type=antibody_group,antibody_group_method=escape,antibody_path=data/antibodies.csv,antigen_embeddings_path=embeddings/antigen_embeddings.pt,antigen_embedding_granularity=residue,antigen_embedding_type=difference,antibody_embedding_granularity=sequence,antibody_embedding_type=concatenation,antibody_embeddings_path=embeddings/antibody_embeddings.pt 

