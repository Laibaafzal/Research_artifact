import os
import json
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)

print("✅ CUDA Available:", torch.cuda.is_available())


with open('complete.json', encoding='utf-8') as f:
    data = json.load(f)

tokens = [entry["tokens"] for entry in data]
labels = [entry["labels"] for entry in data]

label_dict = {
    "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6, "B-DATE": 7, "I-DATE": 8,
    "B-ACT": 9, "I-ACT": 10, "B-EMO": 11, "I-EMO": 12,
    "B-SPIRIT": 13, "I-SPIRIT": 14, "B-TIME": 15, "I-TIME": 16,
    "B-PRO": 17, "B-ROLE": 18, "I-ROLE": 19
}
id2label = {v: k for k, v in label_dict.items()}

def preprocess_data(example):
    example["labels"] = [label_dict[label] for label in example["labels"]]
    return example

dataset = Dataset.from_dict({"tokens": tokens, "labels": labels})
dataset = dataset.map(preprocess_data)

split1 = dataset.train_test_split(test_size=0.1, seed=42)
train_val_dataset = split1["train"]
test_dataset = split1["test"]

split2 = train_val_dataset.train_test_split(test_size=0.1111, seed=42) 
train_dataset = split2["train"]
val_dataset = split2["test"]

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_dict),
    id2label=id2label,
    label2id=label_dict,
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        is_split_into_words=True,
        max_length=512,
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx >= len(labels):
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
val_tokenized = val_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized = test_dataset.map(tokenize_and_align_labels, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=4,
    push_to_hub=False,
    report_to="none",
    resume_from_checkpoint=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_labels = [[l for l in label if l != -100] for label in labels]
    true_preds = [
        [p for p, l in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(preds, labels)
    ]
    flat_preds = sum(true_preds, [])
    flat_labels = sum(true_labels, [])
    return {"accuracy": accuracy_score(flat_labels, flat_preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


trainer.train()

save_path = "./deberta-NER"
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

results = trainer.evaluate(test_tokenized)
print("✅ Final Test Accuracy:", results["eval_accuracy"])

torch.cuda.empty_cache()
print("🎉 All done!")
