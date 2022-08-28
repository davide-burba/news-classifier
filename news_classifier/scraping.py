import requests
from bs4 import BeautifulSoup
import time
import sys
from dataclasses import dataclass


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


def get_scraper(scraper_class="FrontiersScraper", scraper_params={}):
    return getattr(sys.modules[__name__], scraper_class)(**scraper_params)


@dataclass
class FrontiersScraper:
    """Scrape news data from Frontiers Website"""

    n_pages: int = 3

    def run(self):
        """Retrieve blog articles (titles and urls) from frontiers blog by category.

        Returns:
            Dict[str,List[Dict[str,Any]]]
        """
        data = {}
        for category in FRONTIERS_BLOG_CATEGORIES:
            print(category)
            data[category] = self._retrieve_blog_articles(
                category=category, n_pages=self.n_pages
            )
        return data

    def _retrieve_blog_articles(self, category="life-science", n_pages=2):
        assert category in FRONTIERS_BLOG_CATEGORIES

        data = []
        for page in range(1, n_pages + 1):
            print(f"page {page}/{n_pages}")
            url = f"https://blog.frontiersin.org/category/{category}/page/{page}"

            response = requests.get(url)
            if not response.ok:
                print(response.reason)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for div in soup.find_all("div", class_="mh-excerpt"):
                anchor = list(div.children)[1]
                data.append(
                    {
                        "article_url": anchor.get("href"),
                        "article_title": anchor.get("title"),
                        "page": page,
                    }
                )

            # to be respectful
            time.sleep(0.5)

        return data
