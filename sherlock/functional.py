from multiprocessing import Pool
import multiprocessing
import os
import random
import re

from collections import OrderedDict
from datetime import datetime
from functools import partial
from typing import Union, Tuple

from tqdm import tqdm
import pyarrow.lib
import pandas as pd
import numpy as np
import ast
import string

core_count = multiprocessing.cpu_count()  #1
size = 100
#spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

from functional import pseq, seq
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.word_embeddings import extract_word_embeddings_features
from sherlock.features.custom_features import extract_addl_feats
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features
from sherlock.features.helpers import literal_eval_as_str, keys_to_csv
from sherlock.global_state import is_first, set_first, reset_first
ignoreList = ['#na','#n/a','na','n/a','none','nan','blank','blanks']

def as_py_str(x):
    return x.as_py() if isinstance(x, pyarrow.lib.StringScalar) else x

def to_string_list(x):
    return literal_eval_as_str(x, none_value="")

def random_sample(values: list):
    random.seed(13)
    return random.sample(values, k=min(1000000, len(values)))

def special_token_repl(text: str):
    replaced_text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    replaced_text = re.sub(string=replaced_text,pattern=' +',repl=' ')
    return replaced_text

# Clean whitespace from strings by:
#   * trimming leading and trailing whitespace
#   * normalising all whitespace to spaces
#   * reducing whitespace sequences to a single space

def normalise_whitespace(data):
    if isinstance(data, str):
        return re.sub(r"\s{2,}", " ", data.strip())
    else:
        return data

def additional_processing(value):

    #print('Additional Processing:',value)
    if value is None or pd.isnull(value) and str(value).lower() in ignoreList:
      return_val = ''
    else:
      value = str(value).replace('\xa0',' ').strip()
      return_val = removeASCII(value)

    return return_val

def normalise_string_whitespace(col_values):

    id = col_values[0]
    table_name = col_values[1]
    column_name = col_values[2]

    normalized_values = list(map(normalise_whitespace, col_values[3:]))
    
    ### Removing the table and column name from values ## Added to remove features list
    normalized_values = [val for val in normalized_values if str(val).lower() not in [table_name.lower(),column_name.lower()]]
    
    normalized_values_upd = [id] + [table_name] + [column_name] + normalized_values
    return normalized_values_upd

#### Remove ASCII Characters from the data
def removeASCII(strs):
    return ''.join([char for word in str(strs) for char in word if ord(char)<128])

def extract_features(col_values: list):
   
    """Extract features from raw data.
    """
    vec_dim = 400
    reuse_model=True

    id = col_values[0]
    table_name = col_values[1]
    column_name = col_values[2]
    col_values = col_values[3:]

    n_samples = 1000000
    n_values = len(col_values)
    features = OrderedDict()

    print('Custom Preprocessing started:',datetime.now())
    cleaned_population_nan = pseq(map(additional_processing, col_values), processes=core_count, partition_size=size)
    cleaned_population_nan = list(cleaned_population_nan)
    print('Custom preprocessing completed:',datetime.now())
    uniq_cleaned_population = len(set([val.lower() for val in cleaned_population_nan if len(val)>0]))

    if n_samples < n_values:
      random.seed(13)
      cleaned_sample_nan = random.sample(cleaned_population_nan, k=n_samples)
    else:
      n_samples = n_values
      cleaned_sample_nan = cleaned_population_nan	

    cleaned_sample_wo_nan = [val for val in cleaned_sample_nan if len(val)>0]
    cleaned_sample_wo_nan_uncased = [val.lower() for val in cleaned_sample_wo_nan]
    uniq_cleaned_sample = list(set(cleaned_sample_wo_nan))

    print('*'*100)
    extract_bag_of_characters_features(cleaned_sample_wo_nan, features)
    print('='*100)
    extract_word_embeddings_features(cleaned_sample_wo_nan,features,prefix = 'values')
    print('='*100)
    extract_word_embeddings_features([table_name],features,prefix = 'table')
    print('='*100)
    extract_word_embeddings_features([column_name],features,prefix = 'column')
    print('='*100)
    extract_bag_of_words_features(cleaned_sample_nan,cleaned_sample_wo_nan_uncased, features)
    print('='*100)
    extract_addl_feats(cleaned_sample_nan, features)
    print('*'*100)
            
    # TODO use data_no_null version?
    print('paragraph embedding started..')
    infer_paragraph_embeddings_features(uniq_cleaned_sample, features, vec_dim, reuse_model)
    print('paragraph embedding completed..')
    print('*'*100)
    
    features['table_population'] = n_values
    features['table_sample'] = n_samples
    features['uniq_values_population'] = uniq_cleaned_population

    lengths = [len(s) for s in cleaned_population_nan]
    n_none = sum(1 for _l in lengths if _l == 0)

    features["none-agg-has_population"] = 1 if n_none > 0 else 0
    features["none-agg-percent_population"] = n_none / n_values if n_values >0 else 0
    features["none-agg-num_population"] = n_none
    features["none-agg-all_population"] = 1 if n_none == n_values else 0

    features['id'] = id
    features['table_name'] = table_name
    features['column_name'] = column_name

    print('Completed...')
    print('#'*100)

    return features

def normalise_float(value):
    if isinstance(value, str):
        return value

    return "%g" % value


def values_to_str(values):
    return ",".join(map(normalise_float, values)) + "\n"


def numeric_values_to_str(od: OrderedDict):
    return od.keys(), values_to_str(od.values())


def keys_on_first(key_value_tuple, first_keys_only: bool):
    if first_keys_only:
        if is_first():
            set_first()
            return list(key_value_tuple[0]), key_value_tuple[1]
        else:
            return None, key_value_tuple[1]
    else:
        return list(key_value_tuple[0]), key_value_tuple[1]


# Only return OrderedDict.values. Useful in some benchmarking scenarios.
def values_only(od: OrderedDict):
    return list(od.values())


# Eliminate serialisation overhead for return values. Useful in some benchmarking scenarios.
def black_hole(od: OrderedDict):
    return None


def ensure_path_exists(output_path):
    path = os.path.dirname(output_path)

    if not os.path.exists(path):
        os.makedirs(path)

def parallelize_dataframe(df, func):
    n_cores = multiprocessing.cpu_count()
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def remove_table_column_name(values,table_name,column_name):
    return [val for val in values if str(val).lower() not in [table_name.lower(), column_name.lower()]]    

def extract_features_to_csv(parquet_df):

    start = datetime.now()
    
    features_df = pd.DataFrame()

    parquet_df['clean_values'] = parquet_df.apply(lambda x: remove_table_column_name(x['values'],x['table_name'],x['column_name']),axis=1)   
    data_values = parquet_df['clean_values'].values.tolist()
    
    id = parquet_df['id'].values.tolist()

    parquet_df['column_name'] = parquet_df['column_name'].apply(special_token_repl)
    column_name = parquet_df['column_name'].values.tolist()

    parquet_df['table_name'] = parquet_df['table_name'].apply(special_token_repl)
    table_name = parquet_df['table_name'].values.tolist()

    parquet_values = [ [id[val]] + [table_name[val]] + [column_name[val]] + list(data_values[val]) for val in range(len(parquet_df)) ]     

    normalized_list = pseq(map(normalise_string_whitespace, parquet_values), processes=core_count, partition_size=size)
    features_dict = pseq(map(extract_features, normalized_list), processes=core_count, partition_size=size)
    features_df = pd.DataFrame.from_dict(features_dict)
    
    print('Null Features column:',[col for col in features_df.columns if sum(pd.isnull(features_df[col]))>0])
    features_df = features_df.fillna(0)
    features_df['start_time'] = start
    features_df['end_time'] = datetime.now()
    features_df['execution_time'] = datetime.now() - start

    print(f"Finished. Processed {len(parquet_df)} rows in {datetime.now() - start}")    
    
    return features_df
