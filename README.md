# News Classifier

This project is used to run classification of news articles from [Frontiers](https://www.frontiersin.org) using [huggingface](https://huggingface.co) environment. 

The project is used for both model training and deployment.

The code is easily extensible and can be used as a template for other text classification tasks and beyond.

**Table of Contents**
- [Install the project](#install-the-project)
- [Model training and evaluation](#model-training-and-evaluation)
    - [Run a task](#run-a-task)
    - [Logging](#logging)
    - [Test](#test)
- [Deployment](#deployment)
    - [On a dedicated server](#on-a-dedicated-server)
    - [Serverless (AWS)](#serverless-aws)

## Install the project

**With [poetry](https://python-poetry.org):** `poetry install`

**With [docker](https://www.docker.com):** `docker-compose up -d`

To run a command with docker, preceed the commands described below with `docker-compose exec news_classifier`.

## Model training and evaluation

### Run a task

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
    - The default model is a [distilled Bert Transformer](https://huggingface.co/distilbert-base-cased).
    - During training, the following metrics are monitored: loss, accuracy, macro-precision/recall/f1.
    - All the parameters of the model are logged with mlflow.
- `evaluate`: evaluate the classifier on a dataset (the test set by default). It computes:
    - confusion metrics
    - global metrics: accuracy, macro-precision/recall/f1
    - accuracy/precision/recall/f1 for each class.

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

With docker-compose, an mlflow service is already started and available on port 5000. 
The port can be easily changed in the `docker-compose.yml` file.

### Test

To execute the tests, run: 

```bash
poetry run pytest tests
```

## Deployment

There are two options to deploy the app for inference.

### On a dedicated server

Api exposing a `predict` method, built with [FastAPI](https://fastapi.tiangolo.com).

Run the app with: 
```bash
poetry run uvicorn api.app:app --port <PORT>
```

Once turned on, you can try the api on `localhost:<PORT>/docs`.

When using docker, a service for the api is started and is available on port `5001`.
The port can be easily changed in the `docker-compose.yml` file.

### Serverless (AWS)

Lambda paired with an implicit API, built with [AWS SAM](https://aws.amazon.com/serverless/sam/). 

Code and model are enclosed in the container image. 


To build the app, first copy `news_classifier`, `poetry.lock`, `pyproject.toml`, and the desired `model.p` in `sam-app/app`. This is needed to build the docker image for `sam`. Then run:

```bash
# move into sam-app folder
cd sam-app
# build the image
sam build
# test the app locally
sam local start-api
```

To deploy the app on AWS, run:
```bash
sam deploy
```
