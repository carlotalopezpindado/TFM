import configparser
from sqlalchemy import create_engine, MetaData, Table
from transformers import BitsAndBytesConfig
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

config = configparser.ConfigParser()
config.read('config.ini')

user = config['database']['user']
password = config['database']['password']
host = config['database']['host']
port = config['database']['port']
database = config['database']['database']

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
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    system_prompt = """
    Las personas a las que asistes únicamente hablan el idioma en el que preguntan, por lo que únicamente puedes responder 
    en el mismo idioma que se te haga la pregunta. Esto es muy importante. Si no, no entenderán tus respuestas.
    Estás asistiendo en consultas sobre documentos del Boletín Oficial del Estado (BOE). Responde con información relevante 
    y precisa, utilizando únicamente la información contenida en los documentos disponibles. Si la información no está 
    disponible o la pregunta excede el alcance de tu conocimiento actual, informa al usuario de manera clara y directa, 
    evitando especulaciones o suposiciones. Proporciona una respuesta detallada y específica a la pregunta del usuario, 
    incluyendo referencias al documento o documentos relevantes cuando sea posible.
    Responde con la información más actualizada que tengas acerca de la pregunta.
    """

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

def query_index(query_engine):
    while True:
        query = input("Please enter your query: ")
        response = query_engine.query(query)
        print(response)
        
def main():
    username = input("Username: ")
    password = input("Password: ")

    role = get_user_role(username, password)

    if role is not None:
        print(f"User authenticated with role: {role}")
        if role == 'leg':
            vector_store = FaissVectorStore.from_persist_dir("indexes/legislacion_y_regulacion")
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="indexes/legislacion_y_regulacion")
        
        elif role == 'edu':
            vector_store = FaissVectorStore.from_persist_dir("indexes/educacion_y_cultura")
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="indexes/educacion_y_cultura")
    
        elif role == 'adm':
            vector_store = FaissVectorStore.from_persist_dir("indexes/administracion_publica_y_procedimientos")
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="indexes/administracion_publica_y_procedimientos")
            
        llm = init_llm()
        service_context = ServiceContext.from_defaults(llm = llm, embed_model=embed_model)
        index = load_index_from_storage(storage_context=storage_context, service_context = service_context)
        qe = index.as_query_engine(llm=llm, response_mode="compact")
        
        query_index(qe)
        
    else:
        print("Invalid credentials")

if __name__ == "__main__":
    main()
