import pandas as pd
import configparser
from transformers import BitsAndBytesConfig
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
from huggingface_hub import login

config = configparser.ConfigParser()
config.read('config.ini')

file_path = 'evaluation/qas/qa_adm.csv'
df = pd.read_csv(file_path)

model_name = config['indexing']['indexing_model']
context_window = int(config['indexing']['context_window'])
max_new_tokens = 256
temp = float(config['indexing']['temperature'])
top_k = int(config['indexing']['top_k'])
top_p = float(config['indexing']['top_p'])
embed_model = config['indexing']['embed_model']
hf_key = config['keys']['huggingface']


def init_llm():
    login(hf_key)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

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

    query_wrapper = PromptTemplate(
        f"<s> [INST] {system_prompt} Dada la siguiente pregunta del usuario, proporciona una respuesta detallada y específica utilizando" 
        "la información contenida en los documentos. Responde en el idioma en el que se te realice la pregunta"
        "Asegúrate de incluir referencias al documento o documentos relevantes cuando sea posible.\nPregunta: {query_str} [/INST] </s>\n"
    )

    llm = HuggingFaceLLM(                                                        
        model_name=model_name,                                                 
        tokenizer_name=model_name,                                             
        query_wrapper_prompt=query_wrapper,                            
        context_window=context_window,                                            
        max_new_tokens=max_new_tokens,                                           
        model_kwargs={"quantization_config": quantization_config},      
        generate_kwargs={"do_sample": True, "temperature": temp, "top_k": top_k, "top_p": top_p}, 
        device_map="auto",  
        system_prompt=system_prompt
    )
    return llm

def load_index():
    vector_store = FaissVectorStore.from_persist_dir(f"processing/indexes/adm")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=f"processing/indexes/adm")
    index = load_index_from_storage(storage_context=storage_context, service_context=ServiceContext.from_defaults(llm=llm, embed_model=embed_model))
    return index

def responder(question):
    response = qe.query(question)
    return response

    
llm = init_llm()
    
index = load_index()
qe = index.as_query_engine(llm=llm, response_mode="compact")
            
respuestas = pd.DataFrame()
for index, row in df.iterrows():
    respuesta = responder(row['question'])
    respuestas.append(respuesta)
    print(respuesta)
    print('\n---------------------------------------------------------\n')

output_file_path = 'evaluation/rag_ans/rag_adm.csv'
respuestas.to_csv(output_file_path, index=False)