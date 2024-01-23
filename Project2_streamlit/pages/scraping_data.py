import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

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

def scrape_page(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage: {url}")
            return []
    except Exception as e:
        print(f"Error occurred while fetching the webpage: {url}, Error: {e}")
        return []
 
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []

    for review_block in soup.find_all("div", class_="oa_reactionBlock"):
        review = {}

        review_text = review_block.find("h4", class_="oa_reactionText")
        review['review'] = review_text.text.strip() if review_text else "No text"
        reviews.append(review)

        review['company'] = url

        stars = review_block.find("div", class_="oa_stackLevel")
        review['stars'] = len(stars.find_all("i", class_="fas fa-star active")) if stars else "Pas noté"
 
        author_name = review_block.find("span", itemprop="author")
        review['author'] = author_name.text if author_name else "No author"
 
        comment_date = review_block.find("div", class_="oa_date")
        if comment_date:
            date_str = comment_date.text.strip()
            match = re.search(r'\d{2}/\d{2}/\d{4}', date_str)
            review['date'] = match.group(0) if match else "No date"
        else:
            review['date'] = "No date"
 
        
 
    return reviews

# Scrape the selected URL
reviews = scrape_page(option)

# Creating a DataFrame from the reviews
df_reviews = pd.DataFrame(reviews)

df_reviews['company'] = df_reviews['company'].str.replace(r'^https://www.opinion-assurances.fr/assureur-', '', regex=True)
df_reviews['company'] = df_reviews['company'].str.replace(r'\.html.*$', '', regex=True)
df_reviews['company'] = df_reviews['company'].str.replace('-', ' ', regex=True)

# Display the DataFrame in Streamlit
st.write(df_reviews)