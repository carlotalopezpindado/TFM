import streamlit as st
from sqlalchemy import create_engine, MetaData, Table
import configparser

# Leer configuración
config = configparser.ConfigParser()
config.read('config.ini')

user = config['database']['user']
password = config['database']['password']
host = config['database']['host']
port = config['database']['port']
database = config['database']['database']

# para cuando usemos docker
# DATABASE_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
# para usar en local
DATABASE_URL = f"mysql+pymysql://{user}:{password}@localhost:{port}/{database}"

def get_user_role(username, password):
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    users = Table('users', metadata, autoload_with=engine)
    conn = engine.connect()
    query = users.select().where(users.c.UserName == username).where(users.c.Password == password)
    result = conn.execute(query).fetchone()
    conn.close()
    return result.Rol if result else None

def app():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = get_user_role(username, password)
        if role is not None:
            st.success(f"User authenticated with role: {role}")
            st.session_state['role'] = role
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

    if 'role' in st.session_state:
        st.experimental_set_query_params(page='chatbot')
        st.experimental_rerun()
