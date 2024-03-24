import pandas as pd
import torch
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Document

quantization_config = BitsAndBytesConfig( # configuración de cuantificación del modelo -> reduce tamaño y aumenta velocidad
    load_in_4bit=True,                    # carga el modelo en formato 4bits
    bnb_4bit_compute_dtype=torch.float16, # más configuraciones relacionadas con eso
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(                                                             # se carga mixtral desde hugging face
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",                            # modelo
    tokenizer_name="mistralai/Mixtral-8x7B-Instruct-v0.1",                        # tokenizador
    query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),  # plantilla para envolver las consultas al modelo, lo que ayuda a guiar al modelo para responder de una manera específica
    context_window=3900,                                                          # tamaño máximo de la ventana de contexto que el modelo puede considerar para cada predicción
    max_new_tokens=256,                                                           # tokens que el modelo puede generar
    model_kwargs={"quantization_config": quantization_config},                    # configuración de cuantificación
    generate_kwargs={"temperature": 0.2, "top_k": 5, "top_p": 0.95},              # parámetros para la generación de texto
    device_map="auto",                                                            # el modelo se distribuye automaticamente entre los dispositivos disponibles
    system_prompt="Always respond in Spanish."
)

service_context = ServiceContext.from_defaults(llm = llm, embed_model="BAAI/bge-m3")

df = pd.read_parquet('data_clean.parquet')
dfs = {}  
labels = ["Legislation and Regulation", "Public Administration and Procedures", "Education and Culture", "Economy and Finance"]

for label in labels:
    dfs[label] = df[df['classification_result'] == label]

def build_index(df, service_context):
    docs = [Document(content=text) for text in df['text']]
    vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    
    return vector_index

vi_leg = build_index(dfs['Legislation and Regulation'], service_context)
vi_pub_admin = build_index(dfs['Public Administration and Procedures'], service_context)
vi_educ = build_index(dfs['Education and Culture'], service_context)
vi_econ = build_index(dfs['Economy and Finance'], service_context)

vi_leg.storage_context.persist(persist_dir="processing/indexes")
vi_pub_admin.storage_context.persist(persist_dir="processing/indexes")
vi_educ.storage_context.persist(persist_dir="processing/indexes")
vi_econ.storage_context.persist(persist_dir="processing/indexes")

# PARA VOLVER A CARGAR LOS ÍNDICES!!!

# from llama_index.core import StorageContext, load_index_from_storage
# rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
# load index
# index = load_index_from_storage(storage_context)

'''
"Contexto: El usuario está consultando sobre documentos específicos del Boletín Oficial del Estado 
(BOE) clasificados en una de las siguientes categorías: Legislación y Regulación, Administración Pública y Procedimientos, 
Educación y Cultura, Economía y Finanzas. Basado en la categoría y el contenido del documento, responde en español con 
información relevante y precisa. Considera las restricciones de acceso según el perfil del usuario, asegurando que la 
información proporcionada sea accesible para el solicitante. Utiliza un tono formal y claro, adecuado para la comunicación 
profesional y legal.

Instrucción: [INST] Dada una pregunta del usuario relacionada con uno de los documentos del BOE en la categoría 
{categoría_específica}, proporciona una respuesta detallada y específica utilizando la información contenida en 
los documentos accesibles para el perfil del usuario. Asegúrate de incluir referencias al documento o documentos 
relevantes cuando sea posible. [/INST] </s>\n"

Siempre responde en español.
'''