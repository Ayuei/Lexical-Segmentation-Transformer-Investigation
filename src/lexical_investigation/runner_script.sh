#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

set -e

base_dir=pretrained/
task_name="mednli"
data_dir="/scratch1/ngu143/Super_Transfer/data/"


for folder in $(ls "${base_dir}"/); do
	full_path="${base_dir}""${folder}"
	#transformers bert "${full_path}"/model.ckpt "${full_path}"/bert_config.json "${full_path}"/pytorch_model.bin

	cp "${full_path}"/bert_config.json "${full_path}"/config.json

	python3 ./model.py --vocab_file "${full_path}"/vocab.txt \
		   --init_checkpoint "${full_path}" \
		   --task_name "${task_name}" \
		   --data_dir "${data_dir}"
done
