import argparse
import datasets
import torch
from transformers import Trainer, TrainingArguments,\
    RobertaForSequenceClassification, RobertaTokenizer,\
    ElectraForSequenceClassification, ElectraTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .dataset import load_data
import numpy


def get_preprocess_f(label_mapping, tokenizer):
    def fun(example):
        processed_example = tokenizer(example['utterance'], padding=True, truncation=True)
        processed_example['labels'] = [label_mapping[lbl] for lbl in example['labels']]
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
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
    else:
        train_dataset, dev_dataset = datasets.load_dataset('banking77', split=['train', 'test'])

    if args.model == 'robeczech':
        tokenizer = RobertaTokenizer.from_pretrained('ufal/robeczech-base', max_length=10)
        model = RobertaForSequenceClassification.from_pretrained('ufal/robeczech-base',
                                                                 num_labels=len(all_labels))
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
        num_train_epochs=1,
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        warmup_steps=200,
        weight_decay=0.01,
    )
    train_dataset = train_dataset.map(preprocess_f, batch_size=256, batched=True)
    dev_dataset = dev_dataset.map(preprocess_f,  batch_size=256, batched=True)

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
    predictions = trainer.predict(dev_dataset)
    true = [inv_label_mapping[l] for l in predictions.label_ids]
    pred = [inv_label_mapping[p] for p in numpy.argmax(predictions.predictions, axis=-1)]
    cm = confusion_matrix(true, pred)
    dsp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(inv_label_mapping.values()))
    dsp.plot()
    plt.savefig('cm.png')
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--model', type=str, default='roberta')
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    main(args)
