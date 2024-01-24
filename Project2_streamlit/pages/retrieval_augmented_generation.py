import streamlit as st
import dataset
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_distances
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

st.title('Visualisation de Word2Vec avec PCA et TSNE')

def plot_results(result, words):
    plt.figure(figsize=(12,8))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    st.pyplot(plt)  # Utilisez st.pyplot pour afficher le graphique dans Streamlit



data = dataset.all_data['review_en_lemmatized']

model_w2v = Word2Vec(data, min_count=1)
words = list(model_w2v.wv.key_to_index)

num_words = st.slider('Nombre de mots', min_value=100, max_value=1000, value=100, step=100)

word_vectors = np.array([model_w2v.wv[word] for word in words[:num_words]])

pca = PCA(n_components=2)
result_pca = pca.fit_transform(word_vectors)
tsne = TSNE(n_components=2)
result_tsne = tsne.fit_transform(word_vectors)

method = st.selectbox('Méthode de réduction de dimensionnalité', ('PCA', 'TSNE'))
if method == 'PCA':
    result = result_pca
else:
    result = result_tsne

plot_results(result, words[:num_words])

