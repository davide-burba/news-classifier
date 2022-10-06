# News Classifier

This project is used to run classification of news articles from [Frontiers](https://www.frontiersin.org) using [huggingface](https://huggingface.co) environment. 

The project is used for both model training and deployment.

The code is easily extensible and can be used as a template for other text classification tasks and beyond.

## Install the project

**With [poetry](https://python-poetry.org):** `poetry install`

**With [docker](https://www.docker.com):** *Coming soon*

## Model training and evaluation

The entry-point for the project is the `main.py` file, which is used to execute tasks defined in `./news_classifier/tasks`. 

To run a task:

```bash
poetry run python main.py <task_name> [<optional_config_path>] [<optional_output_dir>]
```


It's easy to add new tasks by inherithing from `news_classifier.tasks.BaseTask`.

The following tasks are defined:
- `scrape_data`: scrape data from [Frontiers blog page](https://blog.frontiersin.org).
- `clean_data`: remove articles not scraped correctly
- `format_data`: apply transformations and split in train/valid/test
- `train`: train and validate a text classifier (with [huggingface](https://huggingface.co))
- `evaluate`: evaluate the classifier

Each of the task can be configured via a `yaml` configuration file. You can check the code documentation to see how to configure it. A default one is provided in `config.yml`.

In the default configuration categories are merged in 2 macro-categories (`HEALTH` and `OTHER`). On the test set it achieves about 80% on all the metrics (accuracy, precision, recall).

To run the sequence of tasks with the default configuration and output folder, you can run:

```bash
poetry run python main.py scrape_data
poetry run python main.py clean_data
poetry run python main.py format_data
poetry run python main.py train
poetry run python main.py evaluate
```

### Logging 

Logging is managed with [MLFlow](https://mlflow.org/docs/latest/tracking.html), and results are stored to a local SQLite database. 
To see the mlflow UI you can run:
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Deployment

### On a private server
*Coming soon*

### With AWS
*Coming soon*


## Test

To execute the tests, run: 

```bash
poetry run pytest tests
```