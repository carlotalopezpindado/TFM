import streamlit as st

st.set_page_config(page_title="Chatbot del BOE", layout="centered")

st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Selecciona una página", ["Login", "Chatbot"])

if page == "Login":
    st.experimental_set_query_params(page="login")
    import pages.login as login
    login.app()

if page == "Chatbot":
    st.experimental_set_query_params(page="chatbot")
    import pages.chatbot as chatbot
    chatbot.app()
