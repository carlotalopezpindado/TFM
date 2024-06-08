import pandas as pd
import random
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from gptcache.adapter.langchain_models import LangChainChat
from gptcache import cache
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

openai_key = config['keys']['openain']

def get_msg_func(data, **_):
    return data.get("messages")[-1].content
cache.init(pre_embedding_func=get_msg_func)
cache.set_openai_key()

df = pd.read_parquet('classified_data.parquet')

os.makedirs('temp_texts', exist_ok=True)

def guardar_textos_temporales(df, categoria, n=30):
    textos_categoria = df[df['classification_result'] == categoria]['text'].tolist()
    textos_seleccionados = random.sample(textos_categoria, n)
    archivos = []
    for i, texto in enumerate(textos_seleccionados):
        archivo_path = f'temp_texts/{categoria}_{i}.txt'
        with open(archivo_path, 'w', encoding='utf-8') as file:
            file.write(texto)
        archivos.append(archivo_path)
    return archivos

text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=50, chunk_size=2000)
chat = LangChainChat(chat=ChatOpenAI(temperature=0, openai_api_key=openai_key))
chain = QAGenerationChain.from_llm(chat, text_splitter=text_splitter)

def generar_preguntas_respuestas(archivos):
    pares_qa = []
    for archivo in archivos:
        loader = TextLoader(archivo, encoding="utf-8")
        doc = loader.load()
        texts = text_splitter.split_documents(doc)
        fragmento = random.choice(texts)
        qa = chain.invoke(fragmento.page_content)
        if 'questions' in qa and qa['questions']:
            pares_qa.append({'question': qa['questions'][0]['question'], 'answer': qa['questions'][0]['answer']})
    return pares_qa

categorias = df['classification_result'].unique()
preguntas_respuestas = []
for categoria in categorias:
    archivos = guardar_textos_temporales(df, categoria)
    preguntas_respuestas.extend(generar_preguntas_respuestas(archivos))

df_qa = pd.DataFrame(preguntas_respuestas)
df_qa.to_csv('qa.csv', index=False)

import shutil
shutil.rmtree('temp_texts')