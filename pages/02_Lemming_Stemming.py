import streamlit as st
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def run():
    st.header("Lemming & Stemming Page")
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    st.title("Stemming et Lemmatization en NLP")

    text_input = st.text_input("Entrez une phrase à traiter", "Stemming is interesting")

    method = st.selectbox(
        "Choisissez une méthode",
        ["Stemming avec PorterStemmer", "Stemming avec LancasterStemmer", "Stemming avec SnowballStemmer", "Lemmatization"]
    )

    def process_text(text, method):
        tokens = nltk.word_tokenize(text)
        if method != "Lemmatization":
            if method == "Stemming avec PorterStemmer":
                stemmer = PorterStemmer()
            elif method == "Stemming avec LancasterStemmer":
                stemmer = LancasterStemmer()
            elif method == "Stemming avec SnowballStemmer":
                stemmer = SnowballStemmer("english")
            return ' '.join([stemmer.stem(token) for token in tokens])
        else:
            pos_tags = nltk.pos_tag(tokens)
            lemmatizer = WordNetLemmatizer()
            return ' '.join([lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags])

    if text_input:
        st.write("Résultat :")
        st.write(process_text(text_input, method))
