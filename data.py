import json
import random
from collections import Counter, defaultdict

import pandas as pd




# @title JSON to dataframe helper functions
def undersample_df(df, n_times=3):
    class_0,class_1 = df.category_index.value_counts()
    c0 = df[df['category_index'] == 0]
    c1 = df[df['category_index'] == 1]
    df_0 = c0.sample(int(n_times * class_1))
    
    undersampled_df = pd.concat([df_0, c1],axis=0)
    return undersampled_df

def get_ds_synthetic_data(undersample_n=3, min_w=3):
    short_task = {
      "bugzilla": """How to query bugs using the custom fields with the Bugzilla REST API?""",
      "databases": """Which technology should be adopted for the database layer abstraction: Object/Relational Mapping (ORM) or a Java Database Connectivity API (JDBC)?""",
      "gpmdpu": """Can I bind the cmd key to the GPMDPU shortcuts?""",
      "lucene": """How does Lucene compute similarity scores for the BM25 similarity?""",
      "networking": """Which technology should be adopted for the notification system, Server-Sent Events (SSE) or WebSockets?""",
    }

    with open('relevance_corpus.json') as ipf:
        aux = json.load(ipf)
        raw_data = defaultdict(list)
        for d in aux:
            if d['task'] == 'yargs':
                continue

            raw_data['text'].append(d['text'])
            raw_data['question'].append(short_task[d['task']])
            raw_data['source'].append(d['source'])
            raw_data['category_index'].append(1 if d['weight'] > min_w else 0)
            raw_data['weights'].append(d['weight'] if d['weight'] > min_w else 0)
 
        data = pd.DataFrame.from_dict(raw_data)
        data = undersample_df(data, n_times=undersample_n)
        data = data.sample(frac=1).reset_index(drop=True)
      
    return data

def get_class_weights(y, smooth_factor=0, upper_bound=5.0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    clazz = {cls: float(majority / count) for cls, count in counter.items()}
    result = {}
    for key, value in clazz.items():
        if value > upper_bound:
            value = upper_bound
        
        result[key] = value
    return result
    
    
def add_raw_data(result, data, use_pyramid=False):
    s = data['source']
    if 'docs.oracle' in s or 'developer.android' in s:
        source_type = 'api'
    elif 'stackoverflow.com' in s:
        source_type = 'so'
    elif 'github.com' in s:
        source_type = 'git'
    else:
        source_type = 'misc'
    
    if use_pyramid:
        pyramid = data['category_index']
    else:
        pyramid = 1 if data['weights'] > 1 else 0        
    
    result['text'].append(data['text'])
    result['question'].append(data['question'])
    result['source'].append(data['source'])
    result['category_index'].append(pyramid)
    result['weights'].append(data['weights'])
    result['source_type'].append(source_type)
    

# FIXME: simplest change to go from 10-fold to a simple 50% split
# FIXME: this method prioritizes putting tasks that have SO artifacts for the test set
def greedy_stack_overflow_selection(raw_data, target_count=0.2):
    all_tasks = sorted(list(set([d['question'] for d in raw_data])))
    
    all_artifact_task_pairs = list(set([(d['question'], d['source']) for d in raw_data]))

    random.seed(20211015)
    random.shuffle(all_artifact_task_pairs)

    test_tasks_lst = []
    for task_i, source_j in all_artifact_task_pairs:
        if 'stackoverflow.com' in source_j:
            test_tasks_lst.append(task_i)

        if len(test_tasks_lst) == int(target_count * len(all_tasks)):
            break

    return test_tasks_lst

    
    
