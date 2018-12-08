import requests
from bs4 import BeautifulSoup
import os
import re
from selenium import webdriver
import time
from langdetect import detect
from rs_helper.classes.Crawler import Crawler


class MediumCrawler(Crawler):

    def __init__(self, out_path: str, query: str, number_of_scrollings):
        """
        :param url: String (URL to start from)
        :param query: String (Search term for medium)
        :param number_of_scrollings: int (Number of Scrollings to be done on medium.com)
        Crawler for medium articles. Chromedriver required!
        """
        super().__init__(out_path=out_path)
        self.url = "https://medium.com/search?q={}".format(query)
        projec_root = os.path.abspath(os.path.dirname(__file__))
        driver_bin = os.path.join(projec_root, "chromedriver")

        self.driver = webdriver.Chrome(executable_path=driver_bin)
        self.driver.get(self.url)
        self.algorithm = query
        self.scroll_to_end(number_of_scrollings)

        self.validate_url = re.compile(
            r'^(?:http|ftp)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    def scroll_to_end(self, n_scroll):
        """
        :param n_scroll: int (Number of scrolls)
        :return: int (Height of the website body)
        Method to scroll to the end of the website
        """
        pause = 3
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        print(last_height)
        i = 0
        while True:
            print("Scrolling...")
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            print("New Height :: {}".format(new_height))
            # Double Check -> Maybe additional entries were not loaded in 3 seconds.
            if new_height == last_height:
                print("Double Checking!")
                time.sleep(20)
                print("Getting new height")
                new_height_2 = self.driver.execute_script("return document.body.scrollHeight")
                print("Getting new height :: {}".format(new_height_2))
                if new_height_2 == last_height:
                    break
            last_height = new_height
            i += 1
            if n_scroll is not "inf":
                if i == n_scroll:
                    break
        return last_height

    def get_whole_html(self):
        """
        :return: String (HTML of the website)
        """
        return self.driver.page_source

    def crawl_medium_posts(self):
        """
        :return: void
        Method to perform the crawl and extract links and texts from scrolled website.
        Reults will be saved in out/ directory.
        """
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        counter_pdf = 1
        post_link_list: list = list()
        print("Start crawling...")

        html = self.get_whole_html()
        soup = BeautifulSoup(html, "html.parser")
        for post_link in soup.findAll('a', {'data-action': 'open-post'}):
            post_url = post_link.get("href")
            # print(post_url)
            if post_url not in post_link_list:
                if re.match(self.validate_url, post_url) is not None:
                    post_link_list.append(post_url)
                    try:
                        post_website = requests.get(post_url)
                        post_text = post_website.text
                        post_soup = BeautifulSoup(post_text, "html.parser")
                        heading = post_soup.find_all("h1", class_="graf--title")
                        if len(heading) > 0:
                            heading = heading[0].get_text()

                            # Abbildungen und deren Titel raus
                            for fig in post_soup.find_all("figure"):
                                fig.decompose()

                            # Pre Tags are code elements in Medium
                            for code in post_soup.find_all("pre"):
                                code.decompose()

                            content_div = post_soup.find_all('div', {"class": 'postArticle-content'})
                            if len(content_div) > 0:
                                temp_soup = BeautifulSoup(content_div[0].text, "html.parser")
                                text = temp_soup.get_text(" ")
                                lang = detect(text)
                                if lang == "en":
                                    regex = '[A-Z]*[a-z]*'
                                    r = re.compile(regex)
                                    findings = r.findall(heading)

                                    file = open(self.out_path+"/{}_{}.txt".format(self.algorithm, counter_pdf,
                                                                                  " ".join(findings).replace("  ",
                                                                                                             " ").strip()),
                                                "w+")
                                    file.write(text)
                                    print("File {} saved!".format(counter_pdf))
                                    counter_pdf += 1
                    except:
                        print("Site could not be found or Link is incorrect!")


if __name__ == "__main__":
    print("NOTE: Chrome window should stay active when scrolling to ensure functionality.")
    algorithms = ["Frequent Pattern Mining", "Frequent Pattern", "Sequential Analytics",
                  "Clustering", "Classification", "Regression", "Association rule", "Sequence Analytics"]
    for i in algorithms:
        i = i.replace(" ", "%20")
        crawler = MediumCrawler(number_of_scrollings="inf",
                                query=i,
                                out_path="out/{}/".format(i))
        crawler.crawl_medium_posts()
