#!/bin/sh

set -e

export CUDA_VISIBLE_DEVICES=3

checkpoint_path=./models/pretrained/uncased_L-12_H-768_A-12/
init_checkpoint="${checkpoint_path}"bert_model.ckpt 
input_file=./pubmed_bert_vocab_sub*tfrecord
config_file=models/pretrained/uncased_L-12_H-768_A-12/bert_config.json
output_dir=./models/finetuned/bert_base_pubmed

mkdir -p output_dir

python3 bert/run_pretraining.py --init_checkpoint "${init_checkpoint}" \
	--input_file "${input_file}" \
	--bert_config_file "${config_file}" \
	--output_dir "${output_dir}" \
	--do_train \
	--do_eval \
	--max_sequence_length 512 \
	--num_training_steps 90000
