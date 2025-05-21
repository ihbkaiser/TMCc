#!/bin/bash
for dataset in IMDB
do
    for weight_loss_ECR in 100 120
    do
        for NUM_TOPICS in 70
        do
            python main.py \
                --model ECRTM \
                --weight_ECR $weight_loss_ECR \
                --use_pretrainWE \
                --num_topics $NUM_TOPICS \
                --dataset $dataset \
                --seed 0 \
                --wandb_prj baselines \
                --device cuda:1
        done
    done
done