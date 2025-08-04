from transformers import (
    LongformerTokenizerFast,
    LongformerForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, DatasetDict, load_from_disk
from evaluate import load as load_metric
import numpy as np
import torch
import os
import shutil

# Config
MODEL_CHECKPOINT = "allenai/longformer-base-4096"
OUTPUT_DIR = "./longformer_model_v3"
EPOCHS = 80
BATCH_SIZE = 1
MAX_LEN = 4096
SPLIT_PATH = "saved_split"

# Dataset
if os.path.exists(SPLIT_PATH):
    split = load_from_disk(SPLIT_PATH)
    if "validation" not in split:
        print("⚠️ Folderul salvat nu are subsetul 'validation'. Îl ștergem și reconstruim...")
        shutil.rmtree(SPLIT_PATH)
        dataset = load_dataset("ktgiahieu/maccrobat2018_2020", split="train")
        temp = dataset.train_test_split(test_size=0.2, seed=42)
        val_test = temp["test"].train_test_split(test_size=0.5, seed=42)
        split = DatasetDict({
            "train": temp["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })
        split.save_to_disk(SPLIT_PATH)
else:
    dataset = load_dataset("ktgiahieu/maccrobat2018_2020", split="train")
    temp = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = temp["test"].train_test_split(test_size=0.5, seed=42)
    split = DatasetDict({
        "train": temp["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    split.save_to_disk(SPLIT_PATH)

train_ds = split["train"]
val_ds = split["validation"]
test_ds = split["test"]

# Labels
label_list = sorted(set(
    tag for ex in list(train_ds) + list(val_ds) + list(test_ds)
    for tag in ex["tags"]
))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Tokenizer & Model
tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
model = LongformerForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model.gradient_checkpointing_enable()

# Tokenization
def tokenize_align(ex):
    tokenized = tokenizer(
        ex["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
        max_length=MAX_LEN,
        return_offsets_mapping=True
    )
    labels = []
    word_ids = tokenized.word_ids()
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != prev_word_idx:
            labels.append(label2id[ex["tags"][word_idx]])
        else:
            labels.append(label2id[ex["tags"][word_idx]])
        prev_word_idx = word_idx
    tokenized["labels"] = labels
    tokenized["global_attention_mask"] = [1] + [0] * (len(tokenized["input_ids"]) - 1)
    return tokenized

train_tok = train_ds.map(tokenize_align)
val_tok = val_ds.map(tokenize_align)

# Metrics
metric = load_metric("seqeval")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_preds = [[id2label[p] for p, l in zip(pred, lab) if l != -100]
                  for pred, lab in zip(preds, p.label_ids)]
    true_labs = [[id2label[l] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, p.label_ids)]
    return metric.compute(predictions=true_preds, references=true_labs)

# TrainingArguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="overall_f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(trainer.evaluate())
