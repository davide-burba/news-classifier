import os
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import glob
import pathlib


def get_model_path() -> str:
    """Get path to model file.

    If `PATH_CLASSIFIER` env variable is set, use it.
    Else, take the last run in `../experiments/train`.

    Returns:
        Path to the model pickle file.
    """
    default_path = None

    root = f"{pathlib.Path(__file__).parent.resolve()}/"
    model_paths = glob.glob(f"{root}/../experiments/train/run_*/model.p")
    if model_paths:
        default_path = sorted(model_paths)[-1]
    return os.getenv("PATH_CLASSIFIER", default_path)


def load_model() -> Any:
    """Load the model using get_model_path.

    If not model is available, return None.
    """
    path = get_model_path()
    if path is None:
        return None
    return pd.read_pickle(path)


class Text(BaseModel):
    text: str


app = FastAPI()
model = load_model()


@app.post("/predict")
async def predict(text: Text):
    if model is None:
        return {"message": "No model loaded."}
    # todo: add background task for logging here
    return model.inference(text.text)
