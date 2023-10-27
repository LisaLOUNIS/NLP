import streamlit as st
from transformers import AutoTokenizer

# Fonction pour effectuer la tokenisation
def tokenize_text(model_name, text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return tokens, token_ids

# Fonction principale Streamlit
def run():
    
    st.title("Tokenization Methods by Hugging Face")

    # Sélection du modèle
    model_name = st.selectbox(
        "Select the model",
        ["bert-base-uncased", "gpt2", "t5-small", "roberta-base", "distilbert-base-uncased"],
    )

    # Entrée du texte
    text = st.text_area("Enter text for tokenization", "Hugging Face is creating technology to advance and democratize NLP.")

    # Bouton pour exécuter la tokenisation
    if st.button("Tokenize"):
        tokens, token_ids = tokenize_text(model_name, text)

        # Affichage des tokens et des identifiants de tokens
        st.write("### Tokens")
        st.write(tokens)

        st.write("### Token IDs")
        st.write(token_ids)

if __name__ == "__main__":
    run()
