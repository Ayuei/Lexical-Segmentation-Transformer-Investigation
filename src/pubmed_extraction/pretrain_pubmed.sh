#!/bin/sh

set -e

export CUDA_VISIBLE_DEVICES=1

checkpoint_path=./models/pretrained/uncased_L-12_H-768_A-12/
init_checkpoint="${checkpoint_path}"bert_model.ckpt 
input_file=./data/sentencepiece/sentence_piece_sub_shard*.tfrecord
config_file=bert_config.json
output_dir=./models/finetuned/bert_base_sentencepiece_higher_lr

mkdir -p output_dir

python3 bert/run_pretraining.py \
	--input_file "${input_file}" \
	--bert_config_file "${config_file}" \
	--output_dir "${output_dir}" \
	--do_train \
	--do_eval \
	--max_sequence_length 512 \
	--learning_rate 1e-4
