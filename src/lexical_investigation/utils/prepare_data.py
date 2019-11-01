# Convert all files to csv
# Load this into df
# Turn this into a tensorflow dataset
# Beware that the examples are hardcoded

# example = InputExample(example['idx'].numpy(),
#                                   example['sentence1'].numpy().decode('utf-8'),
#                                   example['sentence2'].numpy().decode('utf-8'),
#                                   str(example['label'].numpy()))

# Enabling cross-compat between old code and newer code

import dill
import pandas as pd
import tensorflow as tf
import glob
from .utils_glue import *

def _decode_dill_generator(processor, data_path):
    for func, _type in [('get_train_examples', "train"),
                        ('get_dev_examples', "vali"),
                        ('get_test_examples', "test")]:
        yield getattr(processor, func)(data_path), _type

def convert_dill_to_tf_data(data_dir, task_name, prefix=""):
    rename_dict = {
            'text_a': 'sentence1',
            'text_b': 'sentence2',
            'guid': 'idx'}

    data_dict = {
            'train': None,
            'vali': None,
            'test': None
    }

    processor = processors[task_name]()

    for examples, _type in _decode_dill_generator(processor, data_dir):
        df = pd.DataFrame([vars(ex) for ex in tqdm(examples, desc='Dataframe generator')])

        df[['text_a', 'text_b', 'label', 'guid']] = df[['text_a', 'text_b', 'label', 'guid']].astype(str)
        df['text_a'] = df['text_a'].str.lower()
        df['text_b'] = df['text_b'].str.lower()
        df.rename(columns=rename_dict, inplace=True)

        slices = tf.data.Dataset.from_tensor_slices(dict(df)) # Data should fit in memory
        data_dict[_type] = slices

    return data_dict, processor.get_labels()
