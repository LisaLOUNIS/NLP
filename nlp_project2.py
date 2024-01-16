import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt

def scrap(url):
  review_titles = []
  review_dates = []
  review_ratings = []
  review_texts = []
  page_number = []

  soup = BeautifulSoup(requests.get(url).text, "html.parser")
  total_pages = (soup.find('a', {'name': 'pagination-button-last'}))
  if total_pages:
    total_pages = int(total_pages.get_text())
  else:
    total_pages = 1

  for i in range(1, total_pages + 1):
    response = requests.get(f"{url}?page={i}")
    web_page = response.text
    soup = BeautifulSoup(web_page, "html.parser")
    for review in soup.find_all(class_ = "styles_reviewCardInner__EwDq2"):
      review_title = review.find(class_ = "typography_heading-s__f7029 typography_appearance-default__AAY17")
      review_titles.append(review_title.getText())

      review_date = review.select_one(selector="time")
      review_dates.append(review_date.getText())

      review_rating = review.find(class_ = "star-rating_starRating__4rrcf star-rating_medium__iN6Ty").findChild()
      review_ratings.append(review_rating["alt"])

      review_text = review.find(class_ = "typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn")
      if review_text == None:
          review_texts.append("")
      else:
          review_texts.append(review_text.get_text(separator=' '))
    
  return pd.DataFrame(list(zip(review_titles, review_dates, review_ratings, review_texts)), columns =['review_title', 'review_date', 'review_rating', 'review_text'])

def get_links(url):
    soup = BeautifulSoup(requests.get(url).text, "html.parser")

    divs = soup.find_all('div', class_= "paper_paper__1PY90 paper_outline__lwsUX card_card__lQWDv card_noPadding__D8PcU styles_wrapper__2JOo2")

    links = []
    for div in divs:
        anchors = div.find_all('a', href=True)
        for a in anchors:
            href = a['href']
            if href.startswith("/review/"):
                links.append(href)

    links = ["https://www.trustpilot.com" + link for link in links]