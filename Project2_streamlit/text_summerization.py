# summary_model.py

from transformers import pipeline

# Initialisation et configuration du modèle de résumé
def get_summarizer():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

# Fonction pour générer un résumé
def summarize_text(summarizer, text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']
