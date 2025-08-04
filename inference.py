from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from termcolor import colored
import numpy as np

# === Config ===
MODEL_DIR = "./longformer_model_v2/checkpoint-16000"
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
THRESHOLD = 0.01

# === Load model + tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# === Load label mapping ===
label_list = list(model.config.id2label.values())

def predict_entities(text, score_threshold=THRESHOLD):
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                        truncation=True, max_length=4096, is_split_into_words=False)
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits[0]
    predictions = torch.argmax(logits, dim=-1).tolist()
    scores = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()

    # === Reconstruct entities ===
    entities = []
    current_entity = None
    current_text = ""
    current_score_sum = 0
    current_count = 0

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue

        label_id = predictions[i]
        label = label_list[label_id]
        score = scores[i]

        if label == "O":
            if current_entity:
                entities.append({
                    "word": current_text,
                    "label": current_entity,
                    "score": current_score_sum / current_count
                })
                current_entity = None
                current_text = ""
                current_score_sum = 0
                current_count = 0
            continue

        label_clean = label.replace("B-", "").replace("I-", "")

        if current_entity is None or label.startswith("B-") or label_clean != current_entity:
            if current_entity:
                entities.append({
                    "word": current_text,
                    "label": current_entity,
                    "score": current_score_sum / current_count
                })
            current_entity = label_clean
            current_text = text[start:end]
            current_score_sum = score
            current_count = 1
        else:
            current_text += text[start:end]
            current_score_sum += score
            current_count += 1

    if current_entity:
        entities.append({
            "word": current_text,
            "label": current_entity,
            "score": current_score_sum / current_count
        })

    # === Print entities ===
    print("🔍 Detected entities:\n")
    for ent in entities:
        if ent["score"] >= score_threshold:
            color = "green" if ent["score"] > 0.8 else "yellow" if ent["score"] > 0.5 else "red"
            print(colored(f"{ent['word']} -> {ent['label']} ({ent['score']:.2f})", color))


if __name__ == "__main__":
    sentence = "A 45-year-old female presented with persistent cough, fever, and chest pain. She has a history of asthma and was prescribed vitamin D."
    predict_entities(sentence)