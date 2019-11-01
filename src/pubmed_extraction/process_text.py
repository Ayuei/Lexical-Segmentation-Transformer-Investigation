#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install scispacy
#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_sm-0.2.3.tar.gz
#!pip install nltk
#!pip install tensorflow

import os
import re
import sys
import json
import nltk
import random
import logging
import sentencepiece as spm
import scispacy


# In[2]:

import spacy
SENT = "The study was carried out in a borough of London where a there is a disparity of wealth and a large ethnic minority population and therefore may be different to many other areas of the UK. However, the findings have similarities to those of other UK studies examining the patient perspective of provision and uptake of care (17,19), which agree that GPs could provide more information and be more proactive in respect of preconception care provision. In addition, we found that GPs were more likely to provide preconception care to women with medical conditions, and this targeted approach highlighted issues similar to those found by Mortagy et al. (46) who interviewed GPs and secondary care health professionals focusing on women with diabetes."


# In[4]:


nlp = spacy.load("en_core_sci_sm")


# In[5]:


nlp(SENT.lower())


# In[6]:


import tqdm


# In[7]:


from tqdm import tqdm
total_lines = 15809286 # counted via wc -l
import multiprocessing as mp
import uuid

NUM_WORKERS = 19

def worker(job_q):
    with open(f"data/{uuid.uuid4()}", "w+") as file_obj:
        while True:
            document = job_q.get()

            if document is None:
                break

            document = re.sub(r'\n+', '\n', document.lower()).strip()
            processed_text = nlp(document)
            for sent in processed_text.sents:
                file_obj.write(sent.text.strip()+'\n')
            file_obj.write('\n')

job_queue = mp.Queue(maxsize=NUM_WORKERS)

pool = mp.Pool(NUM_WORKERS, initializer=worker, initargs=(job_queue,))

with open(f"{sys.argv[1]}", encoding='utf-8') as inp:
    document = []
    for paragraph in tqdm(inp, total=total_lines):
        if paragraph == '\n': #new document
            job_queue.put('\n'.join(document))
            document = []
        else:
            document.append(paragraph)

    for _ in range(NUM_WORKERS):
        job_queue.put(None)

    pool.close()
    pool.join()


# In[ ]:


#MODEL_PREFIX = "tokenizer" #@param {type: "string"}
#VOC_SIZE = 32000 #@param {type:"integer"}
#SUBSAMPLE_SIZE = 12800000 #@param {type:"integer"}
#NUM_PLACEHOLDERS = 256 #@param {type:"integer"}

#SPM_COMMAND = ('--input={} --model_prefix={} '
#               '--vocab_size={} --input_sentence_size={} '
#               '--shuffle_input_sentence=true ' 
#               '--bos_id=-1 --eos_id=-1').format(
#               PRC_DATA_FPATH, MODEL_PREFIX, 
               #VOC_SIZE - NUM_PLACEHOLDERS, SUBSAMPLE_SIZE)

#spm.SentencePieceTrainer.Train(SPM_COMMAND)

