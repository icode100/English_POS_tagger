import pandas as pd
import json
from collections import defaultdict,Counter

with open('input/train.json','r') as f:
    json_data = f.read()
    

dataset = json.loads(json_data)
for item in dataset:
    item['sentence'].insert(0,'<s>')
    item['sentence'].append('</s>')
    item['labels'].insert(0,'<s>')
    item['labels'].append('</s>')
print(f'length of dataset is {len(dataset)}')

state_tags = defaultdict(set)
sentences = list()
state_seq = list()
vocab = set()
tags = set()
tag_count = defaultdict(int)
word_count = defaultdict(int)
tag_zipped_words = list()
from copy import deepcopy
for item in dataset:
    sentences.append(item['sentence'])
    state_seq.append(item['labels'])
    tag_zipped_words.append(deepcopy(list(zip(item['sentence'],item['labels']))))
    for word,count in list(Counter(item['sentence']).items()):
        word_count[word]+=count
    for tag,count in list(Counter(item['labels']).items()):
        tag_count[tag]+=count   
    vocab = vocab.union(set(item['sentence']))
    tags = tags.union(set(item['labels']))
    for i,word in enumerate(item['sentence']):
        state_tags[word].add(item['labels'][i])