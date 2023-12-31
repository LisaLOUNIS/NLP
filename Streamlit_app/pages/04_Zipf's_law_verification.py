import streamlit as st
import collections
import matplotlib.pyplot as plt
import numpy as np

def verify_zipf_law(text):
    words = text.split()
    freq_counter = collections.Counter(words)
    sorted_freq = sorted(freq_counter.values(), reverse=True)
    
    ranks = np.arange(1, len(sorted_freq) + 1)
    frequencies = np.array(sorted_freq)
    
    fig, ax = plt.subplots(figsize=(12, 6))  # Crée une nouvelle figure et des axes
    ax.scatter(ranks, frequencies, color='blue')
    ax.set_title("Zipf's Law")
    ax.set_xlabel("Rank of the word")
    ax.set_ylabel("Frequency")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    st.pyplot(fig)  # Passe la figure à Streamlit

def run():
    st.title("Verification of Zipf's Law")
    st.write("This application verifies if Zipf's law holds for given articles.")

    article_1 = st.text_area("Enter first article:", "")
    article_2 = st.text_area("Enter second article:", "")
    article_3 = st.text_area("Enter third article:", "")
    
    if st.button("Verify Zipf's Law"):
        st.write("### Article 1")
        verify_zipf_law(article_1)
        
        st.write("### Article 2")
        verify_zipf_law(article_2)
        
        st.write("### Article 3")
        verify_zipf_law(article_3)

if __name__ == "__main__":
    run()