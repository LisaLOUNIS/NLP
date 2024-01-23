import streamlit as st
import requests
from bs4 import BeautifulSoup


# assureur_auto = ['olivier-assurances-assurance-auto', 'assureur-allianz-assurance-auto']
# assureur_sante = ['mgp-assurance-sante', 'april-assurance-sante']

# base_url = 'https://www.opinion-assurances.fr'

url = "https://www.opinion-assurances.fr/tous-les-assureurs.html"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Trouver tous les éléments <a> avec un attribut 'onclick'
links = soup.find_all('a', onclick=True)

# Extraire les URLs (href) de ces éléments
urls = [link.get('href') for link in links]

# Afficher les URLs
for url in urls:
    st.write(url)
