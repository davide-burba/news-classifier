import pandas as pd
from news_classifier.tasks import BaseTask
from news_classifier import get_scraper


class ScrapeDataTask(BaseTask):
    def run(self):
        scraper = get_scraper(
            self.config["scraper_class"], self.config["scraper_params"]
        )
        data = scraper.run()
        pd.to_pickle(data, f"{self.output_dir}/data.p")
