import streamlit as st
from streamlit_chat import message
from sqlalchemy import create_engine, MetaData, Table
import configparser
from transformers import BitsAndBytesConfig
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
from huggingface_hub import login

config = configparser.ConfigParser()
config.read('config.ini')

user = config['database']['user']
password = config['database']['password']
host = config['database']['host']
port = config['database']['port']
database = config['database']['database']
hf_key = config['keys']['huggingface']
model_name = config['indexing']['indexing_model']
context_window = int(config['indexing']['context_window'])
max_new_tokens = int(config['indexing']['max_new_tokens'])
temp = float(config['indexing']['temperature'])
top_k = int(config['indexing']['top_k'])
top_p = float(config['indexing']['top_p'])
embed_model = config['indexing']['embed_model']

# para cuando usemos docker
DATABASE_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
# para usar en local
# DATABASE_URL = f"mysql+pymysql://{user}:{password}@localhost:{port}/{database}"

def get_user_role(username, password):
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    users = Table('users', metadata, autoload_with=engine)
    conn = engine.connect()
    query = users.select().where(users.c.UserName == username).where(users.c.Password == password)
    result = conn.execute(query).fetchone()
    conn.close()
    return result.Rol if result else None

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

def load_index(rol):
    vector_store = FaissVectorStore.from_persist_dir(f"processing/indexes/{rol}")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=f"processing/indexes/{rol}")
    index = load_index_from_storage(storage_context=storage_context, service_context=ServiceContext.from_defaults(llm=st.session_state.llm, embed_model=embed_model))
    return index

if 'llm' not in st.session_state:
    st.session_state.llm = init_llm()

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'

    if st.session_state['page'] == 'login':
        show_login()
    elif st.session_state['page'] == 'chat':
        show_chat()

def show_login():
    st.title("Bienvenido!")
    st.write("Por favor, ingrese su nombre de usuario y contraseña")

    username = st.text_input("Nombre de usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Iniciar sesión"):
        role = get_user_role(username, password)
        if role is not None:
            index = load_index(role)
            st.session_state['query_engine'] = index.as_query_engine(llm=st.session_state.llm, response_mode="compact")
            st.session_state['page'] = 'chat'
            st.rerun()
        else:
            st.error("Nombre de usuario o contraseña incorrecto")

def show_chat():
    st.title("BOEBot")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    query_engine = st.session_state['query_engine']

    for message_data in st.session_state['messages']:
        if message_data["role"] == "user":
            message(message_data["content"], is_user=True)
        else:
            bot_content = message_data['content'].replace('\n', '<br>')
            bot_message = f"""
            <div style="color: black; background-color: #fafafa; padding: 10px; border-radius: 10px;">
                <p>{bot_content}</p>
            </div>
            """
            st.markdown(bot_message, unsafe_allow_html=True)

    user_query = st.text_input("Introduzca su consulta:", key="query_input")

    if st.button("Consultar"):
        if user_query:
            st.session_state['messages'].append({"role": "user", "content": user_query})
            st.experimental_rerun()

    if st.session_state['messages'] and st.session_state['messages'][-1]["role"] == "user":
        user_query = st.session_state['messages'][-1]["content"]
        with st.spinner("Generando respuesta..."):
            response = query_engine.query(user_query)
            st.session_state['messages'].append({"role": "bot", "content": str(response)})

            metadata = response.metadata
            first_url = next(iter(metadata.values()))['filename']
            st.session_state['messages'].append({"role": "bot", "content": f'Para más información, consulte: [aquí]({first_url})'})

        st.experimental_rerun()

if __name__ == "__main__":
    main()