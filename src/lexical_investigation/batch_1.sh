#!/bin/sh

set -e

task_name=${1}

./run_model.sh ${task_name} bert_tf2/pretrained/bert_base_add_vocab test_model 
