import pandas as pd
from news_classifier.tasks import BaseTask
from news_classifier import get_formatter


class FormatDataTask(BaseTask):
    def run(self):
        formatter = get_formatter(
            self.config["formatter_class"], self.config["formatter_params"]
        )
        train, valid, test = formatter.run()
        pd.to_pickle(train, f"{self.output_dir}/train.p")
        pd.to_pickle(valid, f"{self.output_dir}/valid.p")
        pd.to_pickle(test, f"{self.output_dir}/test.p")
