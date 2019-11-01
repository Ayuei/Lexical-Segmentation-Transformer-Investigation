#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import random
import logging
import sentencepiece as spm

import spacy
nlp = spacy.load("en_core_sci_sm")

MODEL_PREFIX = "tokenizer" #@param {type: "string"}
VOC_SIZE = 1000 #@param {type:"integer"}
SUBSAMPLE_SIZE = 20000000 #@param {type:"integer"}
NUM_PLACEHOLDERS = 0 #@param {type:"integer"}
PRC_DATA_FPATH = "data/incomplete_pubmed_sent.txt"

SPM_COMMAND = ('--input={} --model_prefix={} '
               '--vocab_size={} --input_sentence_size={} '
               '--shuffle_input_sentence=true '
               '--bos_id=-1 --eos_id=-1 --num_threads=16').format(
               PRC_DATA_FPATH, MODEL_PREFIX,
               VOC_SIZE - NUM_PLACEHOLDERS, SUBSAMPLE_SIZE)

spm.SentencePieceTrainer.Train(SPM_COMMAND)
