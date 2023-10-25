import importlib
import streamlit as st

# Import dynamique du module Home
Home = importlib.import_module("Home")

# Import dynamique des modules dans le dossier 'pages'
tokenization = importlib.import_module("pages.01_Tokenizers")
lemming_stemming = importlib.import_module("pages.02_Lemming_Stemming")
text_similarity = importlib.import_module("pages.03_Text_Similarity")
zipf_law_verification = importlib.import_module("pages.04_Zipf's_law_verification")

PAGES = {
    "Home": Home,
    "Tokenization": tokenization,
    "Lemming and Stemming": lemming_stemming,
    "Text Similarity": text_similarity,
    "Zipf's law verification" : zipf_law_verification
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Appel dynamique de la page (module) sélectionnée
page = PAGES[selection]
page.run()  # Utilisation de .run() au lieu de .app()
