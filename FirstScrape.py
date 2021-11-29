#using beautiful soup to scrape your first website! (Web Scraping with Python, Mitchell)

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/page1.html')
bs = BeautifulSoup(html.read(), 'html.parser')
print(bs.div)
#note that with line 8 we can print html, head, title, body, h1, or div. Check the site itself to see which HTML elements you want
# note also that we can target nested elements seperating them with a '.'. EX: bs.html.body.h1