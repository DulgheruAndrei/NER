import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Încarcă modelul
MODEL_DIR = "./longformer_model_v3/checkpoint-13120"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

# Creează pipeline-ul de NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Funcție pentru interfață
def detect_entities(text):
    results = ner_pipeline(text)
    output = ""
    for ent in results:
        word = ent['word'].replace("##", "")
        label = ent['entity_group']
        score = ent['score']
        output += f"{word} ({label}: {score:.2f})\n"
    return output or "⚠️ No entities detected."

# Interfață Gradio
gr.Interface(
    fn=detect_entities,
    inputs=gr.Textbox(lines=4, placeholder="Introdu un text medical..."),
    outputs="text",
    title="NER Medical Demo",
    description="Model antrenat pe MACCROBAT 2018–2020"
).launch()