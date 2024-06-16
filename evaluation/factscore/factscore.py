import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

knowledge_df = pd.read_parquet('classified_data.parquet')
knowledge_df = knowledge_df[knowledge_df['classification_result'] == 'Administración Pública y Procedimientos']
knowledge_texts = knowledge_df['text'].tolist()

eval_df = pd.read_parquet('evaluation/rag_answers/rag_adm.parquet')
eval_texts = eval_df['response'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)

print("modelos cargados")

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
    return embeddings

def compute_similarity(embedding1, embedding2):
    embedding1 = embedding1.unsqueeze(0) if len(embedding1.shape) == 1 else embedding1
    embedding2 = embedding2.unsqueeze(0) if len(embedding2.shape) == 1 else embedding2
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

def compute_factscore(knowledge_texts, eval_texts):
    knowledge_embeddings = [embed_text(text) for text in knowledge_texts]
    factscores = []
    i = 0
    for eval_text in eval_texts:
        print(i)
        i = i+1
        eval_embedding = embed_text(eval_text)
        similarity_scores = [compute_similarity(eval_embedding, ke) for ke in knowledge_embeddings]
        factscore = np.mean(similarity_scores)
        factscores.append(factscore)

    return factscores

factscores = compute_factscore(knowledge_texts, eval_texts)

df = pd.DataFrame
df['factscore'] = factscores

df.to_csv('evaluation/factscore/fs_leg.csv', index=False)