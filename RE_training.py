import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, ClassLabel
from collections import Counter

print("✅ CUDA Available:", torch.cuda.is_available())

model_name = "microsoft/deberta-v3-base"
data_path = "RE.json"
save_dir = "C:\\Users\\Ali Zafar\\Documents\\Sarmad Ali\\Relation_Extraction_Model"
timestamp = "2025-06-23_22-00-00"
model_save_path = os.path.join(save_dir, f"{model_name.split('/')[-1]}_{timestamp}")
os.makedirs(model_save_path, exist_ok=True)

with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

label2id = {
    'AFFECTS': 0, 'ASSOCIATED_WITH': 1, 'BELIEVES_IN': 2, 'FEELS': 3, 'GUIDES': 4,
    'HAPPENED_IN': 5, 'HAPPENED_ON': 6, 'HAS_ROLE': 7, 'HOSTS': 8, 'LOCATED_IN': 9,
    'PERFORMS': 10, 'SPEAKS_WITH': 11, 'NO_RELATION': 12
}
id2label = {v: k for k, v in label2id.items()}

for sample in data:
    sample["label"] = label2id[sample["label"]]


dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
val_test = dataset["test"].train_test_split(test_size=0.5, seed=42)
dataset["validation"] = val_test["test"]
dataset["test"] = val_test["train"]


tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(preprocess, batched=True)

train_labels = [example["label"] for example in dataset["train"]]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = class_weights_tensor.to(device)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model.to(device)


with torch.no_grad():
 model.classifier.weight.data *= class_weights_tensor.unsqueeze(1)


training_args = TrainingArguments(
    output_dir=model_save_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"]
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


trainer.train()


preds = trainer.predict(dataset["test"])
labels = preds.label_ids
predictions = np.argmax(preds.predictions, axis=-1)

print("\n🔍 Classification Report:")
print(classification_report(labels, predictions, target_names=[id2label[i] for i in range(len(label2id))]))


cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=[id2label[i] for i in range(len(label2id))],
            yticklabels=[id2label[i] for i in range(len(label2id))],
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n✅ Model and tokenizer saved to: {model_save_path}")
