# Import des bibliothèques nécessaires
import streamlit as st
import nltk
import spacy
import sentencepiece as spm
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TreebankWordTokenizer, WordPunctTokenizer

def run():
    # Configuration du style de la page
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f0f5;
    }
    .big-font {
        font-size:22px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h1 style='text-align: center; color: #646464;'>Testeur de Tokenizers en NLP</h1>
    """, unsafe_allow_html=True)

    # Téléchargement du modèle Punkt pour NLTK
    nltk.download('punkt')

    # Champ de texte pour l'input utilisateur
    st.markdown("<p class='big-font'>Entrez le texte à tokenizer :</p>", unsafe_allow_html=True)
    text_input = st.text_area("", height=200)

    # Choix du tokenizer
    st.markdown("<p class='big-font'>Choisissez un tokenizer :</p>", unsafe_allow_html=True)
    tokenizer_choice = st.selectbox(
        "",
        ["Word Tokenize", "Whitespace Tokenize", "Treebank Word Tokenize", "WordPunct Tokenize", "Spacy Tokenize", "SentencePiece Tokenize"]
    )

    # Fonction pour effectuer le tokenizing
    def tokenize_text(text, tokenizer):
        if tokenizer == "Word Tokenize":
            return word_tokenize(text)
        elif tokenizer == "Whitespace Tokenize":
            tokenizer = WhitespaceTokenizer()
            return tokenizer.tokenize(text)
        elif tokenizer == "Treebank Word Tokenize":
            tokenizer = TreebankWordTokenizer()
            return tokenizer.tokenize(text)
        elif tokenizer == "WordPunct Tokenize":
            tokenizer = WordPunctTokenizer()
            return tokenizer.tokenize(text)
        elif tokenizer == "Spacy Tokenize":
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [token.text for token in doc]
        elif tokenizer == "SentencePiece Tokenize":
            # Assurez-vous d'avoir un modèle SentencePiece pré-entraîné
            sp = spm.SentencePieceProcessor(model_file='your_model.model')
            return sp.encode(text, out_type=str)

    # Affichage du résultat
    if text_input:
        st.markdown("<h2 style='text-align: center; color: #646464;'>Texte tokenisé :</h2>", unsafe_allow_html=True)
        st.write(tokenize_text(text_input, tokenizer_choice))
