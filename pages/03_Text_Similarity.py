from nltk.metrics import edit_distance, jaccard_distance
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def run():
    st.header("Text Similarity Page")
    st.title("Calculateur de similarité de texte")
    
    st.write("""
        Cette application permet de calculer la similarité entre deux textes en utilisant différentes méthodes.
    """)
    
    text1 = st.text_area("Entrez le premier texte", "")
    text2 = st.text_area("Entrez le deuxième texte", "")
    
    similarity_type = st.selectbox(
        "Choisissez une méthode de calcul de similarité",
        ["Cosine Similarity", "Levenshtein Distance", "Jaccard Distance", "Q-Grams (Jaccard on n-grams)"]
    )
    
    if st.button('Calculer la similarité'):
        
        if similarity_type == "Cosine Similarity":
            vectorizer = CountVectorizer().fit_transform([text1, text2])
            vectors = vectorizer.toarray()
            cos_sim = cosine_similarity(vectors)
            st.write(f"La similarité cosinus est : {cos_sim[0][1]}")
        
        elif similarity_type == "Levenshtein Distance":
            lev_dist = edit_distance(text1, text2)
            st.write(f"La distance de Levenshtein est : {lev_dist}")
        
        elif similarity_type == "Jaccard Distance":
            set1 = set(text1.split())
            set2 = set(text2.split())
            jaccard = jaccard_distance(set1, set2)
            st.write(f"La distance de Jaccard est : {jaccard}")
        
        elif similarity_type == "Q-Grams (Jaccard on n-grams)":
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
            tf_matrix = vectorizer.fit_transform([text1, text2])
            qgram_sim = cosine_similarity(tf_matrix.toarray())
            st.write(f"La similarité q-gram est : {qgram_sim[0][1]}")
