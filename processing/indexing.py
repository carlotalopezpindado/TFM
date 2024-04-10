import pandas as pd
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, Document
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

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

    temp = config['indexing']['temperature']
    top_k = config['indexing']['top_k']
    top_p = config['indexing']['top_p']
    
    llm = HuggingFaceLLM(                                                 # se carga el modelo desde hugging face
        model_name = config['indexing']['indexing_model'],              # modelo
        tokenizer_name = config['indexing']['indexing_model'],          # tokenizador
        query_wrapper_prompt = query_wrapper,                             # plantilla para envolver las consultas al modelo
        context_window = config['indexing']['context_window'],                                            # tamaño máximo de la ventana de contexto que el modelo puede considerar para cada predicción
        max_new_tokens = config['indexing']['max_new_tokens'],                                            # tokens que el modelo puede generar
        model_kwargs = {"quantization_config": quantization_config},      # configuración de cuantificación
        generate_kwargs = {"temperature": temp, "top_k": top_k, "top_p": top_p}, # parámetros para la generación de texto
        device_map = "auto",                                              # el modelo se distribuye automaticamente entre los dispositivos disponibles
        system_prompt = system_prompt
    )
    return llm


def init_save_index(docs, llm, save_dir, english_label):
    spanish_label = labels_dic[english_label]
    Settings.llm = llm
    Settings.embed_model = config['indexing']['embed_model'] 
    
    document_objects = [Document(doc_id=str(i), text=doc) for i, doc in enumerate(docs)]
    print("Indexando")
    vector_index = VectorStoreIndex.from_documents(document_objects)
    print("Guardando")
    vector_index.storage_context.persist(persist_dir=f'{save_dir}/{spanish_label.replace(" ", "_").lower()}')

classified_data_path = config['indexing']['classified_data_path'] 
df = pd.read_parquet(classified_data_path)
grouped = df.groupby('classification_result')

system_prompt = config['indexing']['system_prompt']

labels = config['classification']['labels']

for classification, group in grouped:
    torch.cuda.empty_cache()

    print(classification)
    documents = group['text'].tolist()
    
    llm = initialize_llm(classification, system_prompt)
    init_save_index(documents, llm, 'processing/indexes', classification)