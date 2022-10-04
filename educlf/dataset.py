from csv import reader as csv_reader
import random
from datasets import Dataset

def load_data(fn):
    data = {'labels': [], 'utterance': []}
    train_size = 0.9
    with open(fn, 'rt', newline='', encoding='utf-8') as csvfd:
        datareader = csv_reader(csvfd, delimiter='\t')
        for row in datareader:
            if len(row) != 2:
                continue
            data['labels'].append(row[0])
            data['utterance'].append(row[1].strip('"').lower())
    random.shuffle(data['utterance'])
    random.shuffle(data['labels'])
    train_data = {'labels': data['labels'][:int(len(data['labels']) * train_size)],
                  'utterance': data['utterance'][:int(len(data['labels']) * train_size)]}
    eval_data = {'labels': data['labels'][int(len(data['labels']) * train_size):],
                 'utterance': data['utterance'][int(len(data['labels']) * train_size):]}
    return Dataset.from_dict(train_data), Dataset.from_dict(eval_data)