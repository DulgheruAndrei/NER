from transformers import LongformerTokenizerFast, LongformerForTokenClassification
from datasets import load_from_disk
from seqeval.metrics import classification_report
from seqeval.metrics.v1 import classification_report as report_with_dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import re

# === Config ===
BASE_MODEL_DIR = "./longformer_model_v3"
SPLIT_PATH = "saved_split"
MAX_LEN = 4096
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# === Get latest checkpoint directory ===
def get_latest_checkpoint(base_dir):
    checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return base_dir
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r"\d+", x)[0]))
    return os.path.join(base_dir, checkpoints[-1])

MODEL_DIR = get_latest_checkpoint(BASE_MODEL_DIR)

# === Load tokenizer and model ===
tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_DIR, add_prefix_space=True)
model = LongformerForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# === Load label mappings ===
label2id = model.config.label2id
id2label = model.config.id2label

# === Load test dataset ===
split = load_from_disk(SPLIT_PATH)
test_dataset = split["test"]

# === IOB helper ===
def ensure_iob(tag):
    if tag == "O":
        return "O"
    if not tag.startswith("B-") and not tag.startswith("I-"):
        return "B-" + tag
    return tag

# === Evaluation ===
true_labels = []
pred_labels = []

for example in test_dataset:
    tokens = example["tokens"]
    true_tags = example["tags"]

    encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                         truncation=True, padding="max_length", max_length=MAX_LEN)
    word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=MAX_LEN).word_ids()
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

    pred_seq = []
    true_seq = []
    prev_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx or word_idx >= len(true_tags):
            continue
        pred_tag = id2label.get(predictions[idx], "O")
        true_tag = ensure_iob(true_tags[word_idx])
        pred_seq.append(ensure_iob(pred_tag))
        true_seq.append(true_tag)
        prev_word_idx = word_idx

    pred_labels.append(pred_seq)
    true_labels.append(true_seq)

# === Report ===
print("\n📊 Classification Report:\n")
print(classification_report(true_labels, pred_labels))

# === F1 Bar Plot ===
report_dict = report_with_dict(true_labels, pred_labels, output_dict=True)
labels = []
f1_scores = []

for label, scores in report_dict.items():
    if label in ["micro avg", "macro avg", "weighted avg"]:
        continue
    labels.append(label)
    f1_scores.append(scores["f1-score"])

sorted_indices = np.argsort(f1_scores)[::-1]
labels = np.array(labels)[sorted_indices]
f1_scores = np.array(f1_scores)[sorted_indices]

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, f1_scores)
plt.title("F1 Score per Entity (Test Set)")
plt.ylabel("F1 Score")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("f1_scores_test_set.png")
plt.show()