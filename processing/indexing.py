import pandas as pd
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, Document, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.postgres import PGVectorStore
import torch
import psycopg2
from sqlalchemy import make_url
import configparser

## ---------------- DATOS CONFIGURACIÓN INICIAL -------------------------
config = configparser.ConfigParser()
config.read('config.ini')
temp = config['indexing']['temperature']
top_k = config['indexing']['top_k']
top_p = config['indexing']['top_p']
connection_string = config['indexing']['postgres']
db_name = config['indexing']['db_name']
model_name = config['indexing']['indexing_model']
context_window = config['indexing']['context_window']
max_new_tokens = config['indexing']['max_new_tokens']
embed_model = config['indexing']['embed_model']
classified_data_path = config['indexing']['classified_data_path'] 

## ---------------- CONSTANTES ------------------------------------------
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

labels = ["Legislation and Regulation", "Public Administration and Procedures", "Education and Culture"] 

labels_dic = {
    "Legislation and Regulation": "Regulación y Legislación",
    "Public Administration and Procedures": "Administración Pública y Procedimientos",
    "Education and Culture": "Educación y Cultura"
}

system_prompt = """
    Estas asistiendo a una persona española que no habla inglés, por lo que solo puedes responder en español. 
    Estás asistiendo en consultas sobre documentos del Boletín Oficial del Estado (BOE), clasificados en una de las siguientes categorías: 
    Legislación y Regulación, Administración Pública y Procedimientos o Educación y Cultura. Siempre responde en castellano con información 
    relevante y precisa. Utiliza únicamente la información contenida en los documentos disponibles. Si la información no está disponible o 
    la pregunta excede el alcance de tu conocimiento actual, informa al usuario de manera clara y directa, evitando especulaciones o suposiciones.
    Responde con la información más actualizada que tengas acerca de la pregunta."""

## ---------------------------------------------------------------------
def initialize_llm(english_label, system_prompt):
    spanish_label = labels_dic[english_label]  
    query_wrapper = PromptTemplate(
        f"<s> [INST] [Consulta de {spanish_label}] {system_prompt} Dada la siguiente pregunta del usuario relacionada con la categoría {spanish_label} "
        "de documentos del BOE, proporciona una respuesta detallada y específica utilizando la información contenida en los documentos. "
        "Asegúrate de incluir referencias al documento o documentos relevantes cuando sea posible. Responde siempre en español.\nPregunta: {query_str} [/INST] </s>\n"
    )
    
    llm = HuggingFaceLLM(                                                        
        model_name = model_name,                                                 
        tokenizer_name = model_name,                                             
        query_wrapper_prompt = query_wrapper,                            
        context_window = context_window,                                            
        max_new_tokens = max_new_tokens,                                           
        model_kwargs = {"quantization_config": quantization_config},      
        generate_kwargs = {"temperature": temp, "top_k": top_k, "top_p": top_p}, 
        device_map = "auto",                                              
        system_prompt = system_prompt
    )
    return llm

def init_save_index(document_objects, llm, english_label):
    spanish_label = labels_dic[english_label]
    Settings.llm = llm
    Settings.embed_model = embed_model 
    
    url = make_url(connection_string)
    vector_store = PGVectorStore.from_params(database=db_name, host=url.host, password=url.password, port=url.port, user=url.username, table_name=spanish_label.replace(" ", "_").lower(), embed_dim=768, hybrid_search=True, text_search_config="spanish")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Indexando y guardando")
    VectorStoreIndex.from_documents(document_objects, storage_context=storage_context, show_progress=True)
                
def main():       
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True 
            
    df = pd.read_parquet(classified_data_path)
    grouped = df.groupby('classification_result')

    for classification, group in grouped:
        print(classification)
        documents = group[['url', 'text']].to_dict(orient='records')
        
        llm = initialize_llm(classification, system_prompt)
        
        document_objects = [Document(doc_id=doc['url'], text=doc['text']) for doc in documents]
        init_save_index(document_objects, llm, classification)
        
if __name__ == "__main__":
    main()