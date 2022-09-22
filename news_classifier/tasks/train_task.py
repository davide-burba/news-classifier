import pandas as pd
from news_classifier.tasks import BaseTask
from news_classifier import get_modeller


class TrainTask(BaseTask):
    def run(self):
        modeller = get_modeller(
            self.config["modeller_class"], self.config["modeller_params"]
        )
        modeller.fit()
        pd.to_pickle(modeller, f"{self.output_dir}/model.p")
