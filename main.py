import fire
import mlflow
import pathlib
from typing import Union
from news_classifier.utils import build_output_dir, load_config
from news_classifier.tasks import (
    ScrapeDataTask,
    CleanDataTask,
    FormatDataTask,
    TrainTask,
    EvaluateTask,
)

ROOT = f"{pathlib.Path(__file__).parent.resolve()}/"

TASK_MAP = {
    "scrape_data": ScrapeDataTask,
    "clean_data": CleanDataTask,
    "format_data": FormatDataTask,
    "train": TrainTask,
    "evaluate": EvaluateTask,
}

DEFAULT_OUTPUT_DIR = {
    "scrape_data": f"{ROOT}/data/raw",
    "clean_data": f"{ROOT}/data/cleaned",
    "format_data": f"{ROOT}/data/formatted",
    "train": f"{ROOT}/experiments/train",
    "evaluate": f"{ROOT}/experiments/evaluate",
}

DEFAULT_CONFIG = {
    "scraper_class": "FrontiersScraper",
    "scraper_params": {},
    "cleaner_params": {"path_raw_dir": DEFAULT_OUTPUT_DIR["scrape_data"]},
    "formatter_class": "StandardFormatter",
    "formatter_params": {"path_raw_dir": DEFAULT_OUTPUT_DIR["clean_data"]},
    "modeller_class": "BaseTextClassifier",
    "modeller_params": {"path_data_dir": DEFAULT_OUTPUT_DIR["format_data"]},
    "analyzer_class": "ClassificationAnalyzer",
    "analyzer_params": {"path_model": DEFAULT_OUTPUT_DIR["train"]},
}


def main(
    task_name: str,
    config_path: str = f"{ROOT}/config.yml",
    output_dir: Union[str, None] = None,
    run_name: Union[str, None] = None,
):
    f"""
    Run a task.

    Args:
        task_name (str): The name of the task. Available tasks are: {list(TASK_MAP.keys())}
        config_path (str): Path to the yaml config file
        output_dir (str): Path to the directory where the output of the task will be stored
        run_name (Union[None,str]): Optional run name for mlflow
    """
    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task {task_name}")

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR[task_name]
    output_dir = build_output_dir(output_dir)
    print(f"output will be saved at {output_dir}")

    mlflow.set_experiment(task_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(config_path)
        config = load_config(config_path, DEFAULT_CONFIG)

        task = TASK_MAP[task_name](config, output_dir)
        task.run()


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlruns.db")
    fire.Fire(main)
