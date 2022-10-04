import argparse
import datasets
import torch
from transformers import Trainer, TrainingArguments,\
    RobertaForSequenceClassification, RobertaTokenizer,\
    ElectraForSequenceClassification, ElectraTokenizer

from .dataset import load_data
import numpy


def get_preprocess_f(label_mapping, tokenizer):
    def fun(example):
        processed_example = tokenizer(example['utterance'], padding=True, truncation=True)
        processed_example['labels'] = label_mapping[example['labels']]
        return processed_example

    return fun


def get_preprocess_banking_f(tokenizer):
    def fun(example):
        processed_example = tokenizer(example['text'], padding=True, truncation=True)
        processed_example['labels'] = example['label']
        return processed_example
    return fun


def compute_metrics(predictions):
    predicted = numpy.argmax(predictions.predictions, axis=-1)
    print(predictions.label_ids, predicted)
    acc = numpy.sum((predicted == predictions.label_ids)) / len(predicted)
    random_label_ids = numpy.random.randint(low=0, high=predictions.label_ids.max(), size=predictions.label_ids.shape)
    random_acc = numpy.sum((random_label_ids == predictions.label_ids)) / len(predicted)
    majority_label_ids = numpy.ones(shape=predictions.label_ids.shape) * numpy.argmax(numpy.bincount(predictions.label_ids))
    majority_acc = numpy.sum((majority_label_ids == predictions.label_ids)) / len(predicted)
    result = {'accuracy': acc, 'random': random_acc, 'majority': majority_acc}
    print(result)
    return result


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')
    if args.model in ['robeczech', 'electra']:
        train_dataset, dev_dataset = load_data(args.data_fn)
        all_labels = set(train_dataset['labels'])
        label_mapping = {name: n for n, name in enumerate(all_labels)}
    else:
        train_dataset, dev_dataset = datasets.load_dataset('banking77', split=['train', 'test'])

    if args.model == 'robeczech':
        tokenizer = RobertaTokenizer.from_pretrained('ufal/robeczech-base')
        model = RobertaForSequenceClassification.from_pretrained('ufal/robeczech-base', num_labels=len(all_labels))
        preprocess_f = get_preprocess_f(label_mapping, tokenizer)
    elif args.model == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('ufal/eleczech-lc-small')
        model = ElectraForSequenceClassification.from_pretrained('ufal/eleczech-lc-small', num_labels=len(all_labels))
        preprocess_f = get_preprocess_f(label_mapping, tokenizer)
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_length=512)
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=77) # 77 is banking77 specific
        preprocess_f = get_preprocess_banking_f(tokenizer)
    else:
        raise ValueError(f'Unknown model name "{args.model}"')

    model = model.to(device)
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        do_eval=True,
        num_train_epochs=5,
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        warmup_steps=200,
        weight_decay=0.01,
    )
    train_dataset = train_dataset.map(preprocess_f)
    dev_dataset = dev_dataset.map(preprocess_f)

    trainer = Trainer(
        model,
        train_args,
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate(train_dataset)
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--model', type=str, default='roberta')
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    main(args)
