#!/bin/sh

set -e

export CUDA_VISIBLE_DEVICES=2

init_checkpoint=./models/pretrained/bert_base_add_vocab/model.ckpt-83000
input_file=./512_bert_medical_*tfrecord
config_file=./models/pretrained/bert_base_add_vocab/config.json
output_dir=./models/pretrained/bert_base_add_vocab_512

mkdir -p output_dir

python3 bert/run_pretraining.py --init_checkpoint "${init_checkpoint}" \
	--input_file "${input_file}" \
	--bert_config_file "${config_file}" \
	--output_dir "${output_dir}" \
	--do_train \
	--do_eval \
	--max_sequence_length 512 \
	--num_train_steps 10000
