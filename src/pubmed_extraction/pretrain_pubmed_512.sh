#!/bin/sh

set -e

export CUDA_VISIBLE_DEVICES=0

checkpoint_path=./models/pretrained/bert_base_sentencepiece_higher_lr/model.ckpt-89000
input_file=./512_sentencepiece*tfrecord
config_file=bert_config.json
output_dir=./models/finetuned/bert_base_sentencepiece_higher_lr_512

mkdir -p output_dir

python3 bert/run_pretraining.py --init_checkpoint "${checkpoint_path}" \
	--input_file "${input_file}" \
	--bert_config_file "${config_file}" \
	--output_dir "${output_dir}" \
	--do_train \
	--do_eval \
	--learning_rate 1e-4 \
	--max_seq_length 512 \
	--num_train_steps 10000
