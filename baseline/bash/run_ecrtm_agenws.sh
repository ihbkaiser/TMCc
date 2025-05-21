#!/bin/bash
for dataset in AGNews
do
    for weight_loss_ECR in 20 30 40 50
    do
        for NUM_TOPICS in 10 20 70
        do
            python main.py \
                --model ECRTM \
                --weight_ECR $weight_loss_ECR \
                --use_pretrainWE \
                --num_topics $NUM_TOPICS \
                --dataset $dataset \
                --seed 0 \
                --wandb_prj baselines
        done
    done
done
