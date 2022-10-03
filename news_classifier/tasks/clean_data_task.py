import os
import pandas as pd
from news_classifier.tasks import BaseTask
from news_classifier.cleaning import Cleaner


class CleanDataTask(BaseTask):
    def run(self):

        cleaner = Cleaner(**self.config["cleaner_params"])
        data = cleaner.run()
        pd.to_pickle(data, f"{self.output_dir}/data.p")
