#!/bin/sh

task_name=${1}
model_path=${2}
output=${3}
CUDA_DEVICE=${4}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

python3 main.py --model_type bert \
       --model_name_or_path ${model_path} \
       --task_name ${task_name} \
       --output_dir ./${output} \
       --max_seq_len 64 \
       --do_train \
       --do_eval \
       --do_test \
       --do_lower_case \
       --num_train_epochs 3 \
       --data_dir ~/scratch1/Super_Transfer/data/ \
       --overwrite_output_dir 
