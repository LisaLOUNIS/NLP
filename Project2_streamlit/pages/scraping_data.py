import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

url_all = "https://www.opinion-assurances.fr/tous-les-assureurs.html"

response = requests.get(url_all)
soup = BeautifulSoup(response.text, 'html.parser')

links = soup.find_all('a', onclick=True)

link_dict = {link.get_text(): 'https://www.opinion-assurances.fr' + link.get('href') for link in links if link.get('href').startswith('/assureur')}

urls = [link.get('href') for link in links if link.get('href').startswith('/assureur')]

full_urls = ['https://www.opinion-assurances.fr' + url for url in urls]

# Créer un menu déroulant dans Streamlit
option = st.selectbox(
    'Choisissez une URL',
    list(link_dict.keys()))

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

def scrape_nb_pages(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pagination_div = soup.find('div', {'class': 'table oa_pagination'})

    if pagination_div is None:
        return 1

    right_div = pagination_div.find('div', {'class': 'oa_right'})
    if right_div is None:
        return 1

    last_page_link = right_div.find('a', {'title': 'Allez à la dernière page'})
    if last_page_link is None:
        return 1

    href = last_page_link.get('href')
    match = re.search(r'page(\d+).html', href)
    if match is None:
        return 1

    nb_pages = int(match.group(1))
    return nb_pages


# Scrape the selected URL
reviews = scrape_page(link_dict[option])
st.write(link_dict[option])

nb_pages = scrape_nb_pages(link_dict[option])

page = st.selectbox('Choisissez une page', [i for i in range(1, nb_pages + 1)])

# Scrape the selected page
url = f"{link_dict[option]}?page={page}"
reviews = scrape_page(url)

# Creating a DataFrame from the reviews
df_reviews = pd.DataFrame(reviews)

df_reviews['company'] = df_reviews['company'].str.replace(r'^https://www.opinion-assurances.fr/assureur-', '', regex=True)
df_reviews['company'] = df_reviews['company'].str.replace(r'\.html.*$', '', regex=True)
df_reviews['company'] = df_reviews['company'].str.replace('-', ' ', regex=True)

# Display the DataFrame in Streamlit
st.write(df_reviews)