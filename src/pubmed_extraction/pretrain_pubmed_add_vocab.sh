#!/bin/sh

set -e

export CUDA_VISIBLE_DEVICES=2

checkpoint_path=./models/pretrained/uncased_L-12_H-768_A-12/
init_checkpoint="${checkpoint_path}"bert_model.ckpt 
input_file=./data/sub_*.tfrecord
config_file="${checkpoint_path}"bert_config.json
output_dir=./models/pretrained/bert_base_add_vocab_30k_steps

mkdir -p output_dir

python3 bert/run_pretraining.py --init_checkpoint "${init_checkpoint}" \
	--input_file "${input_file}" \
	--bert_config_file "${config_file}" \
	--output_dir "${output_dir}" \
	--do_train \
	--do_eval \
	--num_train_steps 30000
