import time
import sys
from typing import Union, Dict, Any, List
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup


FRONTIERS_BLOG_CATEGORIES = [
    "life-science",
    "health",
    "neuroscience",
    "psychology",
    "environment",
    "robotics-and-ai",
    "engineering",
    "humanities",
    "sustainability",
    "climate-action",
    "open-science-policy",
]


def get_scraper(
    scraper_class: str = "FrontiersScraper", scraper_params: Union[Dict, None] = None
) -> Any:
    if scraper_params is None:
        scraper_params = {}
    return getattr(sys.modules[__name__], scraper_class)(**scraper_params)


@dataclass
class FrontiersScraper:
    """Scrape news data from Frontiers Website (https://blog.frontiersin.org).

    Args:
        n_pages: The number of pages to scrape. Each page contains 50 articles.
    """

    n_pages: int = 3

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve blog articles (titles, urls, abstracts) from frontiers blog page.

        Returns:
            The scraped data by category.
        """
        data = {}
        for category in FRONTIERS_BLOG_CATEGORIES:
            print(category)
            data[category] = self._retrieve_blog_articles(
                category=category, n_pages=self.n_pages
            )
        return data

    def _retrieve_blog_articles(
        self, category: str = "life-science", n_pages: int = 2
    ) -> List[Dict[str, Any]]:
        assert category in FRONTIERS_BLOG_CATEGORIES

        data = []
        for page in range(1, n_pages + 1):
            print(f"page {page}/{n_pages}")
            url = f"https://blog.frontiersin.org/category/{category}/page/{page}"

            # get page
            response = requests.get(url)
            if not response.ok:
                print(response.reason)
                continue

            # parse page
            soup = BeautifulSoup(response.text, "html.parser")
            for div in soup.find_all("div", class_="mh-excerpt"):
                anchor = list(div.children)[1]
                url = anchor.get("href")
                title = anchor.get("title")
                abstract = self.get_article_abstract(url)

                data.append(
                    {
                        "article_url": url,
                        "article_title": title,
                        "page": page,
                        "abstract": abstract,
                    }
                )

            # to be respectful
            time.sleep(0.5)

        return data

    def get_article_abstract(self, url):
        """Download Frontiers article abstract from url.

        Args:
            url: The url to the article.

        Returns:
            The article abstract (first bold paragraph).
        """
        response = requests.get(url)
        if not response.ok:
            print(f"Failed request for {url}, returning an empty string.")
        soup = BeautifulSoup(response.text, "html.parser")
        try:
            abstract = (
                soup.find_all("div", class_="entry clearfix")[0]
                .find_all("strong")[0]
                .contents[0]
                .text
            )
        except:
            print(f"Failed parsing abstract for {url}, returning an empty string.")
            abstract = ""
        return abstract
