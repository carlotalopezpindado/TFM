import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_translate = "Helsinki-NLP/opus-mt-es-en"
tokenizer_translate = MarianTokenizer.from_pretrained(model_name_translate)
model_translate = MarianMTModel.from_pretrained(model_name_translate).to(device)

model_name_classify = "facebook/bart-large-mnli"
tokenizer_classify = AutoTokenizer.from_pretrained(model_name_classify)
model_classify = AutoModelForSequenceClassification.from_pretrained(model_name_classify).to(device)

def translate_text(text):
    translated = tokenizer_translate(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    translated_text = model_translate.generate(**translated)
    return tokenizer_translate.decode(translated_text[0], skip_special_tokens=True)

def classify_text(text, labels):
    inputs = tokenizer_classify(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_classify(**inputs).logits
    scores = torch.softmax(logits, dim=1)
    results = sorted([(label, score.item()) for label, score in zip(labels, scores[0])], key=lambda x: x[1], reverse=True)  # Ajustado para incluir el primer score
    return results[0][0]  

df = pd.read_parquet('data_clean.parquet')

labels = ["Legislation and Regulation", "Public Administration and Procedures", "Education and Culture", "Economy and Finance"]

classification_results = []

for index, row in df.iterrows():
    text_es = row['text'] 
    text_en = translate_text(text_es)
    classification_label = classify_text(text_en, labels)
    classification_results.append(classification_label)

df['classification_result'] = classification_results
df.to_parquet('data_classified.parquet')
