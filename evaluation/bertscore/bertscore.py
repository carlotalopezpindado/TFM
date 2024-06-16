import pandas as pd
from bert_score import score
from transformers import AutoTokenizer, AutoModel

# Cargar los datos
df1 = pd.read_csv('evaluation/qas/qa_leg.csv')
df2 = pd.read_parquet('evaluation/rag_ans/rag_leg.parquet')

original_responses = df1['answer'].tolist()
rag_responses = df2['response'].tolist()

model_name = "microsoft/deberta-xlarge-mnli"

# Calcular BERTScore
P, R, F1 = score(rag_responses, original_responses, lang='es', model_type=model_name)

# Crear el DataFrame con los resultados
df = pd.DataFrame({
    'Original Response': original_responses,
    'RAG Response': rag_responses,
    'Precision': P.numpy(),
    'Recall': R.numpy(),
    'F1': F1.numpy()
})

# Guardar el DataFrame en un archivo CSV
df.to_csv("bert_scores_leg.csv", index=False)