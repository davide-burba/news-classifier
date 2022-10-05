import sys
import os
from dataclasses import dataclass
from typing import Union, Dict, Any
import pandas as pd
import mlflow

from news_classifier.evaluation import (
    compute_metrics,
    compute_metrics_by_class,
    build_confusion_matrix,
    get_fig_heatmap,
)


def get_analyzer(
    analyzer_class: str = "ClassificationAnalyzer",
    analyzer_params: Union[Dict, None] = None,
) -> Any:
    """Factory for analyzer objects.

    Args:
        analyzer_class: The analyzer class.
        analyzer_params: The params for the analyzer class.

    Returns:
        Analyzer object.
    """
    if analyzer_params is None:
        analyzer_params = {}
    return getattr(sys.modules[__name__], analyzer_class)(**analyzer_params)


@dataclass
class BaseAnalyzer:
    path_model_run: str = None
    path_model: str = "experiments/train"
    path_dataset_file: str = None
    dataset: str = "test"

    def load_model_data(self):
        model = self._load_model()
        data = self._load_data(model)
        return model, data

    def _load_model(self):
        if self.path_model_run is None:
            # if not specified take last run model
            last_run = sorted([v for v in os.listdir(self.path_model) if "run_" in v])[
                -1
            ]
            self.path_model_run = f"{self.path_model}/{last_run}/"

        return pd.read_pickle(f"{self.path_model_run}/model.p")

    def _load_data(self, model):
        if self.path_dataset_file is None:
            # if not specified, take dataset from directory used to train/validate
            assert self.dataset in {"train", "valid", "test"}
            self.path_dataset_file = f"{model.path_run_dir}/{self.dataset}.p"
        return pd.read_pickle(self.path_dataset_file)


@dataclass
class ClassificationAnalyzer(BaseAnalyzer):
    """Perform analysis on classification tasks."""

    def run(self, output_dir: str):
        """Run analysis, save results in output_dir and mlflow.

        Args:
            output_dir: The folder where to store artifacts.
        """
        model, data = self.load_model_data()
        predictions = model.predict(data)

        target_preds = pd.concat(
            [
                data["raw_labels"].reset_index(drop=True),
                predictions[["predicted_labels"]],
            ],
            axis=1,
        )

        y_true = target_preds.raw_labels
        y_pred = target_preds.predicted_labels
        labels = y_true.unique()

        metrics = compute_metrics(y_true, y_pred, labels)
        mlflow.log_metrics(metrics)

        fig_path = f"{output_dir}/metrics_by_class.html"
        metrics_by_class = compute_metrics_by_class(y_true, y_pred, labels)
        fig = get_fig_heatmap(
            pd.DataFrame(metrics_by_class).round(3), x_text="Metric", y_text="Category"
        )
        fig.write_html(fig_path)
        mlflow.log_artifact(fig_path)

        fig_path = f"{output_dir}/confusion_matrix.html"
        confusion_matrix = build_confusion_matrix(y_true, y_pred, labels)
        fig = get_fig_heatmap(
            confusion_matrix, x_text="Predicted value", y_text="Real value"
        )
        fig.write_html(fig_path)
        mlflow.log_artifact(fig_path)
