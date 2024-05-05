import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

## ---------------- DATOS CONFIGURACIÓN INICIAL -------------------------
import configparser
config = configparser.ConfigParser()

model_name_classify = config['classification']['model_name_classify']

## -----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_classify = AutoTokenizer.from_pretrained(model_name_classify)
model_classify = AutoModelForSequenceClassification.from_pretrained(model_name_classify).to(device)

def classify_text(text, labels):
    inputs = tokenizer_classify(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device) # Tokenizamos el texto
    with torch.no_grad(): # No calculamos gradientes ya que no vamos a entrenar
        logits = model_classify(**inputs).logits

    scores = torch.softmax(logits, dim=1) # Aplicamos softmax para obtener las probabilidades
    
    results = sorted([(label, score.item()) for label, score in zip(labels, scores[0])], key=lambda x: x[1], reverse=True)
    
    return results[0][0]

clean_data_path = 'preprocessing/scrapping/clean_data.parquet'
df = pd.read_parquet(clean_data_path)

labels = ["Legislación y Regulación", "Administración Pública y Procedimientos", "Educación y Cultura"]

classification_results = []

for index, row in df.iterrows():
    print(index)
    text = row['text'] 
    classification_label = classify_text(text, labels)  
    classification_results.append(classification_label)
    
df['classification_result'] = classification_results
df.to_parquet('classified_data.parquet')