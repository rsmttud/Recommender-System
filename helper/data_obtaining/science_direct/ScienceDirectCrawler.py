import requests
import pandas as pd
import json
import os
from tqdm import tqdm
from helper_functions import find_nth


# TODO: Seperate date in day, month, year
class ScienceDirectCrawler:

    def __init__(self, query: str, out_dir: str, to_be_num: int = 1000):
        """
        :param query: Query that should be searched in science direct (https://dev.elsevier.com/tips/ScienceDirectSearchTips.htm)
        :param out_dir: Directory where the resulting files should be stored
        :param to_be_num: Amount of articles to get crawled (Needs to be x times 100.
        """
        if to_be_num % 100 != 0:
            raise ValueError("Number of results must be 100*x")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.query = query
        self.base_url = u'https://api.elsevier.com/content/search/'
        self.base_abstract_url = u'https://api.elsevier.com/content/abstract/eid/'
        self.base_article_url = u'https://api.elsevier.com/content/article/eid/'
        self.scope = "scidir"
        self.url = self.base_url + self.scope + '?query=' + self.query + '&count=100'
        self._status_code = ""
        self.config = self.__load_config()
        self.to_be_num = to_be_num
        self.results = []
        self.total_nums = 0
        self.num_res = 0
        self.out_dir = out_dir

    def run(self):
        """
        :return: void
         General function to start crawling.
        """
        result = self.__exec_request(self.url)
        self.results = result['search-results']['entry']
        self.num_res += len(self.results)
        self.total_nums = int(result['search-results']['opensearch:totalResults'])

        if len(self.results) != self.to_be_num:
            while self.num_res < self.total_nums:
                for el in result['search-results']['link']:
                    if el['@ref'] == 'next':
                        next_url = el['@href']
                        next_response = self.__exec_request(next_url)
                        self.results += next_response['search-results']['entry']
                        print(len(self.results))
                        if len(self.results) == self.to_be_num:
                            break
                if len(self.results) == self.to_be_num:
                    break
        self.__get_articles()
        # self.get_abstracts()

    def __get_articles(self):
        """
        :return: void
         Makes calls to science direct to receive the article and abstracts based on the article eid number.
        """
        errors = 0
        for i, doc in enumerate(tqdm(self.results)):
            eid = doc["eid"]
            eid_url = self.base_article_url+eid
            eid_response = self.__exec_request(eid_url)
            # print(eid_response)
            # print(eid_response['abstracts-retrieval-response']['item']['bibrecord']['head']['abstracts'])
            if eid_response is not "failed":
                if isinstance(eid_response['full-text-retrieval-response']['originalText'], str):
                    idx = find_nth(eid_response['full-text-retrieval-response']['originalText'], "1 Introduction", 2)
                    idx = idx if idx != -1 else 0
                    self.results[i]["article"] = eid_response['full-text-retrieval-response']['originalText'][idx:]
                else:
                    self.results[i]["article"] = False
                self.results[i]["abstract"] = eid_response['full-text-retrieval-response']['coredata']['dc:description']
            else:
                self.results[i]["article"] = False
                self.results[i]["abstract"] = False
                errors += 1
        print("{} Documents could not be found!".format(errors))

    def __exec_request(self, URL):
        """
        :param URL: URL to request
        :return: dict with results
        This method actually makes the requests and handles the response of the server
        """
        headers = {
            "X-ELS-APIKey": self.config['apikey'],
            "Accept": 'application/json'
        }

        request = requests.get(
            URL,
            headers=headers
        )
        self._status_code = request.status_code

        if request.status_code == 200:
            return json.loads(request.text)
        else:
            return "failed"

    def __load_config(self):
        """
        :return: void
        Loads configuration file in which the necessary APIKey is stored
        """
        file = open("config.json")
        config_file = json.load(file)
        file.close()
        return config_file

    def save_to_file(self):
        """
        :return: void
        Saves all results to files in the supplied output directory
        """
        abs_errors = 0
        art_errors = 0
        for doc in self.results:
            file = open(self.out_dir + "[" + doc['prism:coverDate'][0]['$'] + "] " + doc["dc:title"] + ".txt", "w")
            file.write(self.__convert_authors(doc["authors"]) + " " + doc["dc:title"] + "\n")
            if doc["abstract"]:
                file.write("\n\n______ABSTRACT______\n")
                file.write(doc["abstract"])
            else:
                abs_errors += 1
            if doc["article"]:
                file.write("\n\n______ARTICLE______\n")
                file.write(doc["article"])
            else:
                art_errors += 1
        print("Not saved/found: {} abstracts, {} articles".format(abs_errors, art_errors))

    def __convert_authors(self, author_dict: dict):
        """
        :param author_dict: dict
        :return: String of authors
        Internal function to get the author names of an article
        """
        res = ""
        for i, _ in enumerate(author_dict['author']):
            res += author_dict['author'][i]["given-name"] + " " + author_dict['author'][i]["surname"] + ", "
        return res

    def save_to_dataframe(self):
        """
        :return: DataFrame
        Saves the results to a pandas dataframe
        """
        titles = list()
        dates = list()
        authors = list()
        for doc in self.results:
            titles.append(doc['dc:title'])
            dates.append(doc['prism:coverDate'])
            authors.append(self.__convert_authors(doc['authors']))
        return pd.DataFrame({"title": titles, "dates": dates, "author": authors})


if __name__ == "__main__":
    crawler = ScienceDirectCrawler(query="Title-Abstr-Key%28Clustering+AND+algorithm%29", to_be_num=1000, out_dir="out/")
    crawler.run()
    crawler.save_to_file()
