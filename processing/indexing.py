import pandas as pd
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate, ServiceContext, VectorStoreIndex, Document
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

quantization_config = BitsAndBytesConfig( # configuración de cuantificación del modelo -> reduce tamaño y aumenta velocidad
    load_in_4bit=True,                    # carga el modelo en formato 4bits
    bnb_4bit_compute_dtype=torch.float16, # más configuraciones relacionadas con eso
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

labels_dic = {
    "Legislation and Regulation": "Regulación y Legislación",
    "Public Administration and Procedures": "Administración Pública y Procedimientos",
    "Education and Culture": "Educación y Cultura"
}

def initialize_llm(english_label, system_prompt):
    spanish_label = labels_dic[english_label]  # etiqueta en español
    query_wrapper = PromptTemplate(
        f"<s>[Consulta de {spanish_label}] {system_prompt} Dada la siguiente pregunta del usuario relacionada con la categoría {spanish_label} de documentos del BOE, "
        "proporciona una respuesta detallada y específica utilizando la información contenida en los documentos. "
        "Asegúrate de incluir referencias al documento o documentos relevantes cuando sea posible. Responde siempre en español.\nPregunta: {query_str} [/INST] </s>\n"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = device if device == "cpu" else "auto"
    
    llm = HuggingFaceLLM(                                                 # se carga el modelo desde hugging face
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1",              # modelo
        tokenizer_name = "mistralai/Mixtral-8x7B-Instruct-v0.1",          # tokenizador
        query_wrapper_prompt = query_wrapper,                             # plantilla para envolver las consultas al modelo
        context_window = 4096,                                            # tamaño máximo de la ventana de contexto que el modelo puede considerar para cada predicción
        max_new_tokens = 1024,                                            # tokens que el modelo puede generar
        model_kwargs = {"quantization_config": quantization_config},      # configuración de cuantificación
        generate_kwargs = {"temperature": 0.2, "top_k": 5, "top_p": 0.8}, # parámetros para la generación de texto
        device_map = device_map,                                          # el modelo se distribuye automaticamente entre los dispositivos disponibles
        system_prompt = system_prompt
    )
    return llm


def init_save_index(df, llm, save_dir, english_label):
    spanish_label = labels_dic[english_label]
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="BAAI/bge-m3")
    docs = [Document(content=text) for text in df[df['classification_result'] == english_label]['text']]
    vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    vector_index.storage_context.persist(persist_dir=f'{save_dir}/{spanish_label.replace(" ", "_").lower()}')

df = pd.read_parquet('preprocessing/zeroshot-classification/classified_data.parquet')

system_prompt = (
    "Estás asistiendo en consultas sobre documentos del Boletín Oficial del Estado (BOE), clasificados en una de las siguientes categorías: "
    "Legislación y Regulación, Administración Pública y Procedimientos o Educación y Cultura. Siempre responde en castellano con información "
    "relevante y precisa. Utiliza únicamente la información contenida en los documentos disponibles. Si la información no está disponible o "
    "la pregunta excede el alcance de tu conocimiento actual, informa al usuario de manera clara y directa, evitando especulaciones o suposiciones."
)

labels = ["Legislation and Regulation", "Public Administration and Procedures", "Education and Culture"]
service_context = ServiceContext.from_defaults(embed_model="BAAI/bge-m3")

for label in labels:
    print(label)
    llm = initialize_llm(label, system_prompt)
    init_save_index(df, llm, 'processing/indexes', label)

# PARA VOLVER A CARGAR LOS ÍNDICES!!!

# from llama_index.core import StorageContext, load_index_from_storage
## rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
## load index
# index = load_index_from_storage(storage_context)