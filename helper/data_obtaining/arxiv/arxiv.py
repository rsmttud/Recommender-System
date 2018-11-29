from urllib.request import urlopen
import os
import re
from datetime import datetime
import xml.etree.ElementTree as etree
import pandas as pd


def crawl_arxiv(search_query: str, path_xml: str, start=0, max_results=1000) -> bool:
    """
    The function is a little helper to crawl arxiv according to: https://arxiv.org/help/api/user-manual
    :param search_query: The search query you would enter in the arxiv-search
    :param path_xml: Path where query results are stored as xml
    :param start: The document number where to start
    :param max_results: The last document index which is crawled
    :return: None
    """
    if max_results > 30000:
        raise ValueError("The API supports only queries to a maximum number of 30.000")

    url = "http://export.arxiv.org/api/query?search_query=all:{} &start={}&max_results={}".format(search_query, start,
                                                                                                  max_results)
    data = urlopen(url).read()
    xml = etree.fromstring(data)
    try:
        file = open(path_xml, "w")
        file.write(etree.tostring(xml).decode("utf-8"))
        file.close()
        return True
    except:
        return False


def transform_arxiv_to_txt(xml_path: str, out_dir: str) -> bool:
    """
    Function creates for every single entry a separate txt file filled with the abstract/summary
    :param xml_path: path to crawled arxiv conform xml file
    :param out_dir: path where all txt will be stored
    :return: bool when every summary was stored as .txt
    """

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        print("Out dir doesnt exist it will be created.")
    namespaces = {'ns0': 'http://www.w3.org/2005/Atom',
                  "ns1": "http://a9.com/-/spec/opensearch/1.1/",
                  "ns2": "http://arxiv.org/schemas/atom"}
    xml = etree.parse(xml_path)
    counter = 0

    for entry in xml.findall("ns0:entry",namespaces):
        summary = entry.find("ns0:summary", namespaces)
        counter += 1
        file_name = str(counter) + ".txt"
        file = open(os.path.join(out_dir, file_name), "w")
        text = summary.text.rstrip("\n")
        file.write(re.sub(' +', ' ', text)) # remove multiple spaces
        file.close()


def transform_arxiv_to_df(xml_path: str) -> pd.DataFrame:
    """
    Transforms a valid arxiv xml into a DataFrame object. The datafame contains: title, year, author
    :param xml_path:
    :return:
    """
    namespaces = {'ns0': 'http://www.w3.org/2005/Atom',
                  "ns1": "http://a9.com/-/spec/opensearch/1.1/",
                  "ns2": "http://arxiv.org/schemas/atom"}

    titles = []
    years = []
    months = []
    days = []
    authors = []
    xml = etree.parse(xml_path)
    for entry in xml.findall("ns0:entry", namespaces):
        sub_authors = []
        for author in entry.findall("ns0:author",namespaces):
            sub_authors.append(author.find("ns0:name", namespaces).text)
        authors.append(sub_authors)
        titles.append(entry.find("ns0:title",namespaces).text)
        datetime_object = datetime.strptime(entry.find("ns0:published",namespaces).text, '%Y-%m-%dT%H:%M:%SZ')
        years.append(datetime_object.year)
        months.append(datetime_object.month)
        days.append(datetime_object.day)
    return pd.DataFrame({"title": titles, "year": years, "month": months, "day": days, "author": authors})
