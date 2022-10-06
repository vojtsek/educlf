import json
import os.path
import pickle

import numpy
import torch
from transformers import Trainer, TrainingArguments,\
    RobertaModel, ElectraModel,\
    RobertaForSequenceClassification, ElectraForSequenceClassification,\
    RobertaTokenizer, ElectraTokenizer, AutoTokenizer,\
    AutoModel, EvalPrediction


def get_preprocess_f(label_mapping, tokenizer):
    def fun(example, return_tensors=False):
        processed_example = tokenizer(example['utterance'],
                                      padding=True,
                                      truncation=True,
                                      return_tensors='pt' if return_tensors else None)
        if 'labels' in example:
            processed_example['labels'] = [label_mapping[lbl] for lbl in example['labels']]
        return processed_example
    return fun


def compute_metrics(predictions):
    predicted = numpy.argmax(predictions.predictions, axis=-1)
    acc = numpy.sum((predicted == predictions.label_ids)) / len(predicted)
    random_label_ids = numpy.random.randint(low=0, high=predictions.label_ids.max(), size=predictions.label_ids.shape)
    random_acc = numpy.sum((random_label_ids == predictions.label_ids)) / len(predicted)
    majority_label_ids = numpy.ones(shape=predictions.label_ids.shape) * numpy.argmax(numpy.bincount(predictions.label_ids))
    majority_acc = numpy.sum((majority_label_ids == predictions.label_ids)) / len(predicted)
    result = {'accuracy': acc, 'random': random_acc, 'majority': majority_acc}
    print(result)
    return result


class IntentClassifierModel:

    def __init__(self, model_description, device, label_mapping, out_dir):
        self._load_mappings(label_mapping)
        if model_description is not None:
            model, tokenizer, preprocess_f = self._load_model_from_desc(model_description, device, label_mapping)
        else:
            model, tokenizer, preprocess_f = None, None, None
        self.model_description = model_description
        self.device = device
        self.out_dir = out_dir
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_f = preprocess_f
        self.trainer = None

    def _load_mappings(self, label_mapping):
        if label_mapping is not None:
            self.lbl2id = label_mapping
            self.id2lbl = {v: k for k, v in label_mapping.items()}
        else:
            self.lbl2id, self.id2lbl = None, None

    @staticmethod
    def _load_model_from_desc(model_description, device, label_mapping):
        model_cls = AutoModel
        if model_description == 'robeczech':
            model_description = 'ufal/robeczech-base'
            model_cls = RobertaForSequenceClassification
            tokenizer_cls = RobertaTokenizer
        elif model_description == 'eleczech':
            model_description = 'ufal/eleczech-lc-small'
            model_cls = ElectraForSequenceClassification
            tokenizer_cls = ElectraTokenizer
        else:
            with open(os.path.join(model_description, 'config.json'), 'rt') as fd:
                cfg = json.load(fd)
            arch = cfg['architectures'][0]
            if 'Roberta' in arch:
                model_cls = RobertaForSequenceClassification
                tokenizer_cls = RobertaTokenizer
            else:
                model_cls = ElectraForSequenceClassification
                tokenizer_cls = ElectraTokenizer
        model = model_cls.from_pretrained(model_description, num_labels=len(label_mapping))
        model = model.to(device)
        tokenizer = tokenizer_cls.from_pretrained(model_description, max_length=128)
        preprocess_f = get_preprocess_f(label_mapping, tokenizer)
        return model, tokenizer, preprocess_f

    def train(self, train_dataset, dev_dataset):
        if self.out_dir is None:
            print('Tried to train but out dir is not specified!')
            return
        train_dataset = train_dataset.map(self.preprocess_f, batch_size=256, batched=True)
        dev_dataset = dev_dataset.map(self.preprocess_f, batch_size=256, batched=True)
        train_args = TrainingArguments(
            output_dir=self.out_dir,
            do_eval=True,
            num_train_epochs=8,
            evaluation_strategy='epoch',
            per_device_train_batch_size=8,
            warmup_steps=200,
            weight_decay=0.01,
        )
        self.trainer = Trainer(
            self.model,
            train_args,
            data_collator=None,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()

    def predict_from_dataset(self, dataset):
        if self.trainer is None:
            print('The model has not been trained yet!')
            return EvalPrediction([], [])
        dataset = dataset.map(self.preprocess_f, batched=True, batch_size=256)
        return self.trainer.predict(dataset)

    def predict_example(self, example):
        if not isinstance(example, list):
            example = [example]
        example = {'utterance': example}
        example = self.preprocess_f(example, return_tensors=True)
        with torch.no_grad():
            feed = {k: v.to(self.model.device) for k, v in example.items()}
            logits = self.model(**feed).logits.cpu().numpy()
        predicted_id = numpy.argmax(logits, axis=-1)
        return [self.id2lbl[pred] for pred in predicted_id]

    def save(self):
        self.trainer.save_model(self.out_dir)
        with open(os.path.join(self.out_dir, 'labels.pkl'), 'wb') as fd:
            pickle.dump(self.lbl2id, fd)

    def load_from(self, save_dir):
        with open(os.path.join(save_dir, 'labels.pkl'), 'rb') as fd:
            label_mapping = pickle.load(fd)
        self._load_mappings(label_mapping)
        self.model, self.tokenizer, self.preprocess_f = self._load_model_from_desc(save_dir,
                                                                                   self.device,
                                                                                   label_mapping)
