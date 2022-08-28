import fire
import mlflow
import pandas as pd
import pathlib
import yaml

from news_classifier import get_scraper, get_formatter
from news_classifier.utils import build_output_dir


ROOT = f"{pathlib.Path(__file__).parent.resolve()}/"
DEFAULT_CONFIG = {
    "scraper_class": "FrontiersScraper",
    "scraper_params": {},
    "formatter_class": "StandardFormatter",
    "formatter_params": {},
}


def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path is not None:
        with open(config_path, "r") as f:
            config.update(yaml.safe_load(f))
    return config


class Main:
    def scrape_data(
        self, config_path=f"{ROOT}/config.yml", output_dir=f"{ROOT}/data/raw/"
    ):
        mlflow.set_experiment("scrape_data")
        mlflow.log_artifact(config_path)

        config = load_config(config_path)
        scraper = get_scraper(config["scraper_class"], config["scraper_params"])
        data = scraper.run()
        output_dir = build_output_dir(output_dir)

        mlflow.log_param("output_dir", output_dir)
        pd.to_pickle(data, f"{output_dir}/data.p")
        print(f"output saved at {output_dir}")

    def format_data(
        self, config_path=f"{ROOT}/config.yml", output_dir=f"{ROOT}/data/formatted/"
    ):
        mlflow.set_experiment("format_data")
        mlflow.log_artifact(config_path)

        config = load_config(config_path)
        formatter = get_formatter(config["formatter_class"], config["formatter_params"])
        train, valid, test = formatter.run()

        output_dir = build_output_dir(output_dir)
        mlflow.log_param("output_dir", output_dir)
        pd.to_pickle(train, f"{output_dir}/train.p")
        pd.to_pickle(valid, f"{output_dir}/valid.p")
        pd.to_pickle(test, f"{output_dir}/test.p")
        print(f"output saved at {output_dir}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlruns.db")
    fire.Fire(Main)
