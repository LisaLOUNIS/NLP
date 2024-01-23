import streamlit as st
import requests
from bs4 import BeautifulSoup

url_all = "https://www.opinion-assurances.fr/tous-les-assureurs.html"

response = requests.get(url_all)
soup = BeautifulSoup(response.text, 'html.parser')

links = soup.find_all('a', onclick=True)

urls = [link.get('href') for link in links if link.get('href').startswith('/assureur')]

full_urls = ['https://www.opinion-assurances.fr' + url for url in urls]

# Créer un menu déroulant dans Streamlit
option = st.selectbox(
    'Choisissez une URL',
    full_urls)

st.write('Vous avez sélectionné l\'URL:', option)