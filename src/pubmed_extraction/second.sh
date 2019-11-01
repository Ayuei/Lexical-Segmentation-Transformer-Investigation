#!/bin/sh

set -e

processed_training_data_dir="shards/sub-shards"
vocab="vocab_builder/medline_vocab_sentencepiece.txt"

files=(sub_shardag)

for file in ${files[@]}; do
	echo Processing "${file}"
	nohup python3 bert/create_pretraining_data.py --input_file "${processed_training_data_dir}"/"${file}" \
		--output_file sentence_piece_"${file}".tfrecord \
		--vocab_file "${vocab}" > sentence_"${file}".out 2>&1 &
done
