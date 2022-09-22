from news_classifier.tasks import BaseTask
from news_classifier import get_analyzer


class EvaluateTask(BaseTask):
    def run(self):
        analyzer = get_analyzer(
            self.config["analyzer_class"], self.config["analyzer_params"]
        )
        analyzer.run(self.output_dir)
