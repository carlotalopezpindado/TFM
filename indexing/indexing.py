import pandas as pd
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, Document, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import configparser
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

## ---------------- DATOS CONFIGURACIÓN INICIAL -------------------------
config = configparser.ConfigParser()
config.read('config.ini')

model_name = config['indexing']['indexing_model']
context_window = int(config['indexing']['context_window'])
max_new_tokens = int(config['indexing']['max_new_tokens'])
temp = float(config['indexing']['temperature'])
top_k = int(config['indexing']['top_k'])
top_p = float(config['indexing']['top_p'])
embed_model = config['indexing']['embed_model']

## ---------------- CONSTANTES ------------------------------------------
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

labels = ["Regulación y Legislación", "Administración Pública y Procedimientos", "Educación y Cultura"] 

system_prompt = """
    Actúa como un asistente especializado en consultas sobre documentos del Boletín Oficial del Estado (BOE). Tu tarea es 
    proporcionar información relevante y precisa basándote únicamente en los documentos disponibles del BOE. Es fundamental 
    que sigas estas instrucciones para asegurar la precisión y relevancia de tus respuestas:

    1. Responde únicamente en el idioma en el que se formule la pregunta. Esto es crucial, ya que las personas a las que asistes 
       solo entienden el idioma en el que preguntan.
    2. Proporciona respuestas detalladas y específicas, utilizando solo la información contenida en los documentos del BOE. 
       No hagas suposiciones ni especulaciones.
    3. Si la información solicitada no está disponible en los documentos del BOE, informa al usuario de manera clara y directa 
       que no tienes la información requerida.
    4. Incluye referencias al documento o documentos relevantes del BOE siempre que sea posible para respaldar tus respuestas.
    5. Utiliza la información más actualizada disponible en los documentos del BOE para responder a las preguntas.

    Aquí hay algunos ejemplos de cómo podrías responder:

    Ejemplo 1:
    Pregunta: ¿Cuál es la última modificación de la Ley de Propiedad Intelectual?
    Respuesta: Según el documento del BOE del [fecha], la última modificación de la Ley de Propiedad Intelectual se realizó el [fecha de modificación], donde se introdujeron cambios en [detalles específicos de los cambios]. Puedes consultar el documento completo en el BOE en la sección [sección del BOE].

    Ejemplo 2:
    Pregunta: ¿Qué establece el Real Decreto 123/2023 sobre el teletrabajo?
    Respuesta: El Real Decreto 123/2023, publicado en el BOE el [fecha], establece las condiciones y requisitos para el teletrabajo, incluyendo [detalles específicos]. Para más información, puedes revisar el documento en la sección [sección del BOE].

    Si no tienes la información requerida, responde de la siguiente manera:
    "Lo siento, no tengo información sobre eso en los documentos disponibles del BOE. Por favor, revisa el BOE directamente o proporciona más detalles para una búsqueda más precisa."

    Siguiendo estas pautas, asegúrate de que cada respuesta sea clara, precisa y útil para el usuario.
"""

embeddings_dimension = 768
## ---------------------------------------------------------------------
def init_llm():
    query_wrapper = PromptTemplate(
    f"<s> [INST] {system_prompt} Dada la siguiente pregunta del usuario, proporciona una respuesta detallada y específica utilizando" 
    "la información contenida en los documentos. Responde en el idioma en el que se te realice la pregunta"
    "Asegúrate de incluir referencias al documento o documentos relevantes cuando sea posible.\nPregunta: {query_str} [/INST] </s>\n"
    )
    
    llm = HuggingFaceLLM(                                                        
        model_name = model_name,                                                 
        tokenizer_name = model_name,                                             
        query_wrapper_prompt = query_wrapper,                            
        context_window = context_window,                                            
        max_new_tokens = max_new_tokens,                                           
        model_kwargs = {"quantization_config": quantization_config},      
        generate_kwargs = {"do_sample": True, "temperature": temp, "top_k": top_k, "top_p": top_p}, 
        device_map = "auto",                                              
        system_prompt = system_prompt
    )
    return llm

def init_save_index(document_objects, llm, label):
    Settings.llm = llm
    Settings.embed_model = embed_model 
    faiss_index = faiss.IndexFlatL2(embeddings_dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    label = label.replace(" ", "_").lower()
    label = label.replace("ó", "o").replace("ú", "u")
    directory = f'processing/indexes/{label}'
    
    index = VectorStoreIndex.from_documents(document_objects, storage_context=storage_context, show_progress=True)
    index.storage_context.persist(directory)
                
def main():
    df = pd.read_parquet('classified_data.parquet')
    grouped = df.groupby('classification_result')

    llm = init_llm()
    
    for classification, group in grouped: 
        documents = group[['url', 'text']].to_dict(orient='records')
        document_objects = [Document(text=doc['text'], metadata={"filename": doc['url']}) for doc in documents]
        
        init_save_index(document_objects, llm, classification)
        
if __name__ == "__main__":
    main()