import pandas as pd
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, Document, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import configparser
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch

## ---------------- DATOS CONFIGURACIÓN INICIAL -------------------------
config = configparser.ConfigParser()
config.read('config.ini')

model_name = config['indexing']['indexing_model']
context_window = int(config['indexing']['context_window'])
max_new_tokens = int(config['indexing']['max_new_tokens'])
temp = config['indexing']['temperature']
top_k = config['indexing']['top_k']
top_p = config['indexing']['top_p']
embed_model = config['indexing']['embed_model']

## ---------------- CONSTANTES ------------------------------------------
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

labels = ["Regulación y Legislación", "Administración Pública y Procedimientos", "Educación y Cultura"] 

system_prompt = """
    Las personas a las que asistes únicamente hablan el idioma en el que preguntan, por lo que siempre tienes que responder en el mismo idioma que se te haga la pregunta. 
    Esto es muy importante. Si no, no entenderán tus respuestas.
    Estás asistiendo en consultas sobre documentos del Boletín Oficial del Estado (BOE), clasificados en una de las siguientes categorías: 
    Legislación y Regulación, Administración Pública y Procedimientos o Educación y Cultura. Siempre responde con información 
    relevante y precisa. Utiliza únicamente la información contenida en los documentos disponibles. Si la información no está disponible o 
    la pregunta excede el alcance de tu conocimiento actual, informa al usuario de manera clara y directa, evitando especulaciones o suposiciones.
    Responde con la información más actualizada que tengas acerca de la pregunta."""

## ---------------------------------------------------------------------
def initialize_llm(label, system_prompt):
    query_wrapper = PromptTemplate(
        f"<s> [INST] [Consulta de {label}] {system_prompt} Dada la siguiente pregunta del usuario relacionada con la categoría {label} "
        "de documentos del BOE, proporciona una respuesta detallada y específica utilizando la información contenida en los documentos. "
        "Asegúrate de incluir referencias al documento o documentos relevantes cuando sea posible.\nPregunta: {query_str} [/INST] </s>\n"
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

def init_save_index(document_objects, llm, label):
    Settings.llm = llm
    Settings.embed_model = embed_model 

    es_client = Elasticsearch(
        hosts=["https://localhost:9200"],
        basic_auth=("elastic", "qTIql*fRyWF*q=6OJBal"),  # Reemplaza con tus credenciales
        verify_certs=False,
        ssl_show_warn=False
    )

    # Use the Elasticsearch client in the vector store
    vector_store = ElasticsearchStore(
        es_client=es_client,
        index_name=label.replace(" ", "_").lower()
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Indexando y guardando")
    VectorStoreIndex.from_documents(document_objects, storage_context=storage_context, show_progress=True)
                
def main():       
    df = pd.read_parquet('classified_data.parquet')
    grouped = df.groupby('classification_result')

    for classification, group in grouped:
        print(classification)
        documents = group[['url', 'text']].to_dict(orient='records')
        
        llm = initialize_llm(classification, system_prompt)
        
        document_objects = [Document(text=doc['text'], metadata = {"filename": doc['url']}) for doc in documents]
        init_save_index(document_objects, llm, classification)
        
if __name__ == "__main__":
    main()