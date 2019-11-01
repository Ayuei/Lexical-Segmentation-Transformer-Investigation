# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
import dill
from tqdm import tqdm
import pandas as pd
import pickle

from utils.utils_glue import DataProcessor, InputExample, InputFeatures

from .utils import DataProcessor, InputExample, InputFeatures
#from ...file_utils import is_tf_available

#if is_tf_available():
#    import tensorflow as tf

def is_tf_available():
    return False

logger = logging.getLogger(__name__)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True,
                                      tokenize=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    #if is_tf_available() and isinstance(examples, tf.data.Dataset):
    #    is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    print(label_map)

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="Iterating"):
        try:
            if ex_index % 10000 == 0:
                logger.info("Writing example %d" % (ex_index))
            if is_tf_dataset:
                example = InputExample(example['idx'].numpy(),
                                example['sentence1'].numpy().decode('utf-8'),
                                example['sentence2'].numpy().decode('utf-8'),
                                example['label'].numpy().decode('utf-8'))

            tokens_a = tokenizer.tokenize(example.text_a) if tokenize else example.text_a
            tokens_b = tokenizer.tokenize(example.text_b) if tokenize else example.text_b
            _truncate_seq_pair(tokens_a, tokens_b, max_length-3)

            inputs = tokenizer.encode_plus(
                tokens_a,
                tokens_b,
                add_special_tokens=True,
                max_length=max_length,
                truncate_first_sequence=True  # We're truncating the first sequence in priority
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    
            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
    
            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(output_mode)
    
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_tokens: %s" % (tokenizer.tokenize(example.text_a+
                    example.text_b)))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  label=label,
                                  tokens_a = tokens_a,
                                  tokens_b = tokens_b))
        except ValueError:
            print("Bad input", example.text_a, example.text_b)
            continue

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features

class QAProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir+"/QA_TrainingSet1.dill", "rb")), "train_trainqa"
            )+self._create_examples(dill.load(open(data_dir+"/QA_TrainingSet2.dill", "rb")),
                    "train_alexa")
            #self._create_examples(
            #dill.load(open(data_dir+"/QA_ValidationSet.dill", "rb")), "dev")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #return self._create_examples(
        #    dill.load(open(data_dir+"/QA_TestSet.dill", "rb")), "test")
        return self._create_examples(
            dill.load(open(data_dir+"/QA_ValidationSet.dill", "rb")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir+"/QA_TestSet.dill", "rb")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, questions, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        def process_text(s, sent_a=True):
            return s

        for qid in questions:
            question_text = questions[qid].text
            for answer in questions[qid].answers:
                guid = "%s-%s" % (set_type, str(qid)+"_"+str(answer.aid))
                answer_text = answer.text

                text_a = process_text(question_text)
                text_b = process_text(answer_text, sent_a=False)

                label = 1 if answer.reference_score > 2 else 0

                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class QuotaProcessor(DataProcessor):
    #"id","qid1","qid2","question1","question2","is_duplicate"
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(open(data_dir + "/train_quora.csv", "rb")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(open(data_dir + "/test_quora.csv", "rb")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(open(data_dir + "/test_quora.csv", "rb")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for _id, row in tqdm(df.iterrows(), desc="Preparing Data", total=len(df)):
            guid = "%s-%s" % (set_type, row["id"])
            text_a = str(row["question1"])
            text_b = str(row["question2"])
            label = int(row["is_duplicate"])
            assert label in self.get_labels()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class RQEProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir + "/train.dill", "rb")), "train"
        )# + \
       # self._create_examples(
       #     dill.load(open(data_dir + "/dev.dill", "rb")), "dev"
       # )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir + "/dev.dill", "rb")), "dev"
        )
        #return self._create_examples(
        #    dill.load(open(data_dir + "/test_labelled.dill", "rb")), "dev"
        #)

    def get_test_examples(self, data_dir):
        """See base class."""
        #return self._create_examples(
        #    dill.load(open(data_dir + "/dev.dill", "rb")), "dev"
        #)
        return self._create_examples(
            dill.load(open(data_dir + "/test_labelled.dill", "rb")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lst, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        lst = lst[0]
        for _id, key in tqdm(enumerate(lst), desc="Preparing Data", total=len(lst)):
            data = lst[key]
            guid = "%s-%s" % (set_type, _id)
            text_a = data["data"][0]
            text_b = data["data"][1]
            label = int(data["label"])
            assert label in self.get_labels()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class DuplicateQuestionsProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                pickle.load(open("./../data/SO_train.pkl", "rb")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                pickle.load(open("./../data/SO_test.pkl", "rb")), "dev"
        )

    def get_labels(self):
        """See base class."""
        return [0,1]

    def _create_examples(self, data, set_type):
        #dict_keys(['orig_title', 'duplicate_title', 'duplicate_id', 'orig_id', 'label'])
        """Creates examples for the training and dev sets."""
        examples = []
        for datum in tqdm(data, desc="parsing"):
            guid = "%s-%s-%s" % (set_type, datum["orig_id"], datum["duplicate_id"])
            text_a = datum['orig_title']
            text_b = datum['duplicate_title']
            label = int(datum['label'])
            assert label in self.get_labels()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class MEDNLIProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir + "/nli_train.dill", "rb")), "train"
        )# + \
        #self._create_examples(
        #    dill.load(open(data_dir + "/nli_dev.dill", "rb")), "dev"
        #)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir + "/nli_dev.dill", "rb")), "dev"
        )
        #return self._create_examples(
        #    dill.load(open(data_dir + "/nli_test_labelled.dill", "rb")), "test"
        #)

    def get_test_examples(self, data_dir):
        """See base class."""
        #return self._create_examples(
        #    dill.load(open(data_dir + "/nli_dev.dill", "rb")), "dev"
        #)
        return self._create_examples(
            dill.load(open(data_dir + "/nli_test_labelled.dill", "rb")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lst, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for _id, data in tqdm(enumerate(lst), desc='Processing NLI'):
            guid = "%s-%s" % (set_type, _id)
            text_a = data['data'][0]
            text_b = data['data'][1]
            label = data['label']
            assert label in self.get_labels()
            examples.append(
               InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SNLIProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir + "/train.dill", "rb")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir + "/dev.dill", "rb")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            dill.load(open(data_dir + "/test.dill", "rb")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral", "-"]

    def _create_examples(self, lst, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for _id, data in tqdm(enumerate(lst), desc='Processing NLI'):
            guid = "%s-%s" % (set_type, _id)
            text_a = data['data'][0]
            text_b = data['data'][1]
            label = data['label']
            assert label in self.get_labels(), f"Label: {label}, data:{data}"
            examples.append(
               InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question1'].numpy().decode('utf-8'),
                            tensor_dict['question2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question'].numpy().decode('utf-8'),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "rqe": 2,
    "quora": 2,
    "so-hard": 2,
    "mednli": 3,
    "snli": 3,
    "mediqa": 2
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "rqe": RQEProcessor,
    "quora": QuotaProcessor,
    "so-hard": DuplicateQuestionsProcessor,
    "mednli": MEDNLIProcessor,
    "snli": SNLIProcessor,
    "mediqa": QAProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "rqe": "classification",
    "quora": "classification",
    "so-hard": "classification",
    "mednli": "classification",
    "snli": "classification",
    "mediqa": "classification",
}
