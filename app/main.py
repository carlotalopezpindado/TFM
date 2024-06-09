import streamlit as st
from streamlit_chat import message
from sqlalchemy import create_engine, MetaData, Table
import configparser

from init_script import llm, index_dict

config = configparser.ConfigParser()
config.read('config.ini')

user = config['database']['user']
password = config['database']['password']
host = config['database']['host']
port = config['database']['port']
database = config['database']['database']

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
            st.session_state['query_engine'] = index_dict[role].as_query_engine(llm=llm, response_mode="compact")
            st.session_state['page'] = 'chat'
            st.rerun()
        else:
            st.error("Nombre de usuario o contraseña incorrecto")

def show_chat():
    st.title("Chatbot - BOE")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    query_engine = st.session_state['query_engine']

    for message_data in st.session_state['messages']:
        if message_data["role"] == "user":
            message(message_data["content"], is_user=True)
        else:
            message(message_data["content"], is_user=False)

    user_query = st.text_input("Introduzca su consulta:", key="query_input")

    if st.button("Consultar"):
        if user_query:
            st.session_state['messages'].append({"role": "user", "content": user_query})
            with st.spinner("Generando respuesta..."):
                response = query_engine.query(user_query)
                st.session_state['messages'].append({"role": "bot", "content": str(response)})
                st.rerun()

if __name__ == "__main__":
    main()