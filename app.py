import streamlit as st
import importlib

# Ajoute un sélecteur pour naviguer entre les différentes pages de l'application
page = st.sidebar.selectbox(
    "Choisir une page",
    ["Home", "Tokenization", "Lemming & Stemming"]
)

# Importe le module Python correspondant à la page sélectionnée
if page == "Home":
    home_module = importlib.import_module("Home")
elif page == "Tokenization":
    token_module = importlib.import_module("pages.01_Tokenizers")
elif page == "Lemming & Stemming":
    lemming_stemming_module = importlib.import_module("pages.02_Lemming_Stemming")
