import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import time

def scrape_page(i):
  page_reviews = []
  time.sleep(2)
  response = requests.get(f"{url}?page={i}")
  web_page = response.text
  soup = BeautifulSoup(web_page, "html.parser")
  if soup=='<html><body>We have received an unusually large amount of requests from your IP so you have been rate limited</body></html>':
    print("request limit")

  for review in soup.find_all(class_ = "styles_reviewCardInner__EwDq2"):
      review_title = review.find(class_ = "typography_heading-s__f7029 typography_appearance-default__AAY17").getText()
      review_date = review.select_one(selector="time").getText()
      review_rating = review.find(class_ = "star-rating_starRating__4rrcf star-rating_medium__iN6Ty").findChild()["alt"]
      review_text = review.find(class_ = "typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn")
      if review_text == None:
          review_text = ""
      else:
          review_text = review_text.get_text(separator=' ')
      page_reviews.append((review_title, review_date, review_rating, review_text))

  return page_reviews

def parallel_scrapping(url):
  soup = BeautifulSoup(requests.get(url).text, "html.parser")
  if soup=='<html><body>We have received an unusually large amount of requests from your IP so you have been rate limited</body></html>':
    print("request limit")
  page_numbers = []
  for a in soup.find('div', class_= "styles_pagination__6VmQv").find_all('a', href=True):
      if(a['name'].startswith('pagination-button-') and a.text.isnumeric()):
          page_numbers.append(int(a.text))
  with concurrent.futures.ThreadPoolExecutor() as executor:
      results = list(executor.map(scrape_page, range(1, max(page_numbers) + 1)))
  flat_results = [item for sublist in results for item in sublist]

  return pd.DataFrame(flat_results, columns =['review_title', 'review_date', 'review_rating', 'review_text'])

def get_links(url):
    soup = BeautifulSoup(requests.get(url).text, "html.parser")

    page_numbers = []
    for a in soup.find('div', class_= "styles_paginationWrapper__fukEb styles_pagination__USObu").find_all('a', href=True):
      if(a['name'].startswith('pagination-button-') and a.text.isnumeric()):
          page_numbers.append(int(a.text))

    divs = soup.find_all('div', class_= "paper_paper__1PY90 paper_outline__lwsUX card_card__lQWDv card_noPadding__D8PcU styles_wrapper__2JOo2")

    links = []

    for i in range(1, max(page_numbers) + 1):
      time.sleep(2)
      response = requests.get(f"{url}?page={i}")
      web_page = response.text
      soup = BeautifulSoup(web_page, "html.parser")
      if soup=='<html><body>We have received an unusually large amount of requests from your IP so you have been rate limited</body></html>':
        print("request limit")
      for div in divs:
          anchors = div.find_all('a', href=True)
          for a in anchors:
              href = a['href']
              if href.startswith("/review/"):
                  links.append(href)

    return ["https://www.trustpilot.com" + link for link in links]

def process_links(url_list):
    df_list = []
    for url in url_list:
        df = parallel_scrapping(url)
        df['URL'] = url
        df_list.append(df)
    result = pd.concat(df_list, ignore_index=True)

    return result


# links for travel insurance company
url = 'https://www.trustpilot.com/categories/travel_insurance_company'
url_list = get_links(url)
df = process_links(url_list)