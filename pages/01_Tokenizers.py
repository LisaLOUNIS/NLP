# Import des bibliothèques nécessaires
import streamlit as st
import nltk
import spacy
import sentencepiece as spm
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TreebankWordTokenizer, WordPunctTokenizer

# Téléchargement du modèle Punkt pour NLTK
nltk.download('punkt')

# Titre de l'application
st.title("Testeur de Tokenizers en NLP")

# Champ de texte pour l'input utilisateur
text_input = st.text_area("Entrez le texte à tokenizer", "")

# Choix du tokenizer
tokenizer_choice = st.selectbox(
    "Choisissez un tokenizer",
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
    st.write("Texte tokenisé :")
    st.write(tokenize_text(text_input, tokenizer_choice))
