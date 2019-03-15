from urllib.request import urlopen
import re
import os
from datetime import datetime
import xml.etree.ElementTree as etree
import pandas as pd
from rs_helper.helper.classes.Crawler import Crawler
import time


class ArxivCrawler(Crawler):

    def __init__(self, out_dir: str, search_query: str, xml_dir: str, start=0, max_results=1000) -> None:
        """
        Crawler to crawl the database ArXiV

        :param out_dir: Output directory
        :type out_dir: str
        :param search_query: https://arxiv.org/help/api/user-manual#query_details
        :type search_query: str
        :param xml_dir: Directory to store the received XML file
        :type xml_dir: str
        :param start: Index to start search from
        :type start: int
        :param max_results: Maximum number of results (default=1000)
        :type max_results: int
        """
        super().__init__(out_dir)

        self.out_dir = out_dir
        self.search_query = search_query
        self.xml_dir = xml_dir
        self.xml_path = None
        self.start = start
        self.max_results = max_results

        self.namespaces = {'ns0': 'http://www.w3.org/2005/Atom',
                           "ns1": "http://a9.com/-/spec/opensearch/1.1/",
                           "ns2": "http://arxiv.org/schemas/atom"}

    def __slugify(self, value: str) -> str:
        """
        Normalizes string, converts to lowercase, removes non-alpha characters,
        and converts spaces to hyphens.

        :param value: String to clean
        :type value: str

        :return: The cleaned token
        :rtype: str
        """
        x = re.sub('[^\w\s-]', '', value).strip().lower()
        x = re.sub('[-\s]+', '-', value)
        x = re.sub('/', "-", x)
        return x

    def save_to_file(self) -> None:
        """
        Function creates for every single entry a separate txt file filled with the abstract/summary
        """

        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
            print("Out_dir doesnt exist it will be created.")

        xml = etree.parse(self.xml_path)
        counter = 0

        for entry in xml.findall("ns0:entry", self.namespaces):
            summary = entry.find("ns0:summary", self.namespaces)
            title = entry.find("ns0:title", self.namespaces)
            year = datetime_object = datetime.strptime(entry.find("ns0:published", self.namespaces).text,
                                                       '%Y-%m-%dT%H:%M:%SZ').year

            counter += 1
            x = "[{}]{}".format(year, title.text) + ".txt"
            file_name = self.__slugify(x)
            file_name = re.sub(" +", " ", file_name)
            file = open(os.path.join(self.out_dir, file_name), "w")
            text = summary.text
            file.write(text.lstrip())  # trim whitespaces on the left side
            file.close()

    def __get_xml(self) -> bool:
        """
        Method to receive the XML

        :return: Status
        :rtype: bool
        """
        if self.max_results > 2000:
            raise ValueError("The API supports only queries to a maximum number of 2000 per call")

        if not os.path.isdir(self.xml_dir):
            os.mkdir(self.xml_dir)
            print("Xml_dir doesnt exist it will be created.")

        url = "http://export.arxiv.org/api/query?search_query={}&start={}&max_results={}".format(self.search_query,
                                                                                                 self.start,
                                                                                                 self.max_results)

        data = urlopen(url).read()
        try:
            xml = etree.fromstring(data)

            t_stamp = str(int(time.time()))
            self.xml_path = os.path.join(self.xml_dir, "arxiv_{}.xml".format(t_stamp))
            file = open(self.xml_path, "w")
            file.write(etree.tostring(xml).decode("utf-8"))
            file.close()
            return True
        except:
            return False

    def save_to_dataframe(self) -> pd.DataFrame:
        """
        Save the crawl results to a DataFrame

        :return: DataFrame of the crawls
        :rtype: pd.DataFrame
        """
        titles = []
        years = []
        months = []
        days = []
        authors = []
        xml = etree.parse(self.xml_path)
        for entry in xml.findall("ns0:entry", self.namespaces):
            sub_authors = []
            for author in entry.findall("ns0:author", self.namespaces):
                sub_authors.append(author.find("ns0:name", self.namespaces).text)
            authors.append(sub_authors)
            titles.append(entry.find("ns0:title", self.namespaces).text)
            datetime_object = datetime.strptime(entry.find("ns0:published", self.namespaces).text, '%Y-%m-%dT%H:%M:%SZ')
            years.append(datetime_object.year)
            months.append(datetime_object.month)
            days.append(datetime_object.day)
        return pd.DataFrame({"title": titles, "year": years, "month": months, "day": days, "author": authors})

    def crawl(self) -> None:
        """
        Actually performs the crawl.

        :return: None
        """
        if self.__get_xml():
            self.save_to_file()
            print("Arvix Crawling - Done!")
        else:
            raise RuntimeError("There is something wrong with the crawled arxiv XML file. Check the path and the xml!")
