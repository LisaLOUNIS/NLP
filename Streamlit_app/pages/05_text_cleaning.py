import streamlit as st
import string
import re
import nltk
from nltk.corpus import stopwords

# Fonction pour initialiser les stopwords
def initialize_stopwords(lang):
    try:
        return set(stopwords.words(lang))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words(lang))

# Fonction pour convertir le texte en minuscules
def convert_to_lowercase(text):
    return text.lower()

# Fonction pour supprimer les mots vides (stop words)
def remove_stop_words(text, stop_words):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

# Fonction pour supprimer la ponctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Fonction pour supprimer les espaces blancs supplémentaires
def trim_extra_whitespace(text):
    return re.sub(' +', ' ', text).strip()

# Fonction principale Streamlit
def run():
    st.title("Text Cleaning Tool")
    
    # Choix de la langue pour les stopwords
    lang_choice = st.selectbox("Choose language for stopwords:", ["english", "french", "german", "spanish"])
    stop_words = initialize_stopwords(lang_choice)
    
    user_input = st.text_area("Paste your text here for cleaning:", "")
    
    options = st.multiselect(
        "Select the cleaning operations to perform:",
        ("Convert to Lowercase", "Remove Stop Words", "Remove Punctuation", "Trim Extra White Space"),
    )
    
    if st.button("Clean Text"):
        cleaned_text = user_input

        if "Convert to Lowercase" in options:
            cleaned_text = convert_to_lowercase(cleaned_text)
        
        if "Remove Stop Words" in options:
            cleaned_text = remove_stop_words(cleaned_text, stop_words)

        if "Remove Punctuation" in options:
            cleaned_text = remove_punctuation(cleaned_text)

        if "Trim Extra White Space" in options:
            cleaned_text = trim_extra_whitespace(cleaned_text)
        
        st.write("### Original Text")
        st.write(user_input)

        st.write("### Cleaned Text")
        st.write(cleaned_text)

if __name__ == "__main__":
    run()
