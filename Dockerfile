FROM python:3.9-slim-buster

RUN mkdir /code
RUN mkdir /code/data
RUN mkdir /code/experiments

COPY poetry.lock /code/
COPY pyproject.toml /code/
COPY main.py /code/main.py
COPY news_classifier /code/news_classifier
COPY api /code/api
COPY config.yml /code/config.yml

WORKDIR /code

RUN apt-get update
RUN apt-get install git -y
RUN pip install poetry; poetry install
CMD poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db