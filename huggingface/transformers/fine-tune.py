from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import evaluate

dataset = load_dataset("D:\.cache\huggingface\datasets\yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=5)
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./out/test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
