FROM public.ecr.aws/lambda/python:3.9

COPY app.py ./
COPY poetry.lock  ./
COPY pyproject.toml  ./
COPY model.p /opt/ml/model.p
COPY news_classifier ./news_classifier

RUN python3.9 -m pip install poetry
RUN poetry config virtualenvs.create false --local
RUN poetry install

CMD ["app.lambda_handler"]
