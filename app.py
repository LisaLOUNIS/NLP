import streamlit as st
from pages import tokenization, lemming_stemming  # Assurez-vous que ces modules sont importables

# Titre principal
st.title("NLP Toolbox")

# Sidebar pour navigation
page = st.sidebar.selectbox(
    "Choose a NLP task",
    ("Home", "Tokenization", "Lemming & Stemming")
)

# Contenu conditionnel
if page == "Home":
    st.write("Welcome to the NLP toolbox. Choose a task from the sidebar.")
elif page == "Tokenization":
    tokenization.run()  # Une fonction 'run' à définir dans votre module 'tokenization'
elif page == "Lemming & Stemming":
    lemming_stemming.run()  # Une fonction 'run' à définir dans votre module 'lemming_stemming'
