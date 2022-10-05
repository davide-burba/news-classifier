from dataclasses import dataclass
import pandas as pd
import sys
import os
from typing import Dict, Union, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    TextClassificationPipeline,
    Trainer,
)
from datasets import Dataset

from news_classifier.evaluation import compute_metrics_huggingface


def get_modeller(
    modeller_class: str = "TextClassifier",
    modeller_params: Union[Dict, None] = None,
) -> Any:
    """Factory for modeller objects.

    Args:
        modeller_class: The name of the modeller class.
        modeller_params: The params for the modeller class.

    Returns:
        The modeller object.
    """
    if modeller_params is None:
        modeller_params = {}
    return getattr(sys.modules[__name__], modeller_class)(**modeller_params)


@dataclass
class TextClassifier:
    """Text classifier model.

    Args:
        path_run_dir: Path to the directory with the data (`train.p`,`valid.p`,`test.p`).
            Data must have a `raw_label` column.
        path_data_dir: Path to the folder with runs for formatted data. Ignored if
            path_run_dir is not None. If path_run_dir is None, take the last run in
            path_data_dir.
        checkpoint: Checkpoint for the huggingface model.
        cache_dir: Cache folder.
        feature_key: Name of the column to be used as input feature.
        truncation: Set to True to truncate feature tokens to a max length.
        max_length: Max length for truncation.
        logging_steps: How often to log validation metrics.
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for training.
        dropout: Dropout level.
    """

    path_run_dir: str = None
    path_data_dir: str = "data/formatted"
    checkpoint: str = "distilbert-base-cased"
    cache_dir: str = "./cache"
    feature_key: str = "article_title"
    truncation: bool = True
    max_length: int = 500
    logging_steps: int = 10
    num_train_epochs: int = 3
    learning_rate: float = 5 * 10e-5
    dropout: float = 0.1

    def fit(self):
        """Train the model."""
        # load data
        train = self.load_data("train")
        valid = self.load_data("valid")

        # convert data to huggingface dataset
        train_dataset = Dataset.from_pandas(train)
        valid_dataset = Dataset.from_pandas(valid)

        # set tokenizers (features,labels)
        self.features_tokenizer = FeaturesTokenizer(
            checkpoint=self.checkpoint,
            cache_dir=self.cache_dir,
            feature_key=self.feature_key,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        self.labels_tokenizer = LabelsTokenizer(train["raw_labels"].values)

        # apply tokenizers (features,labels)
        train_dataset = train_dataset.map(self.features_tokenizer, batched=True)
        train_dataset = train_dataset.map(self.labels_tokenizer, batched=True)
        valid_dataset = valid_dataset.map(self.labels_tokenizer, batched=True)
        valid_dataset = valid_dataset.map(self.features_tokenizer, batched=True)

        # set engine
        self.engine = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=self.labels_tokenizer.num_labels,
            cache_dir=self.cache_dir,
            ignore_mismatched_sizes=True,
            dropout=self.dropout,
        )

        # train
        self.training_args = TrainingArguments(
            f"{self.cache_dir}/trainer_logs_dir",
            evaluation_strategy="steps",
            eval_steps=self.logging_steps,
            logging_steps=self.logging_steps,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
        )
        data_collator = DataCollatorWithPadding(
            tokenizer=self.features_tokenizer.tokenizer
        )
        trainer = Trainer(
            self.engine,
            self.training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            tokenizer=self.features_tokenizer.tokenizer,
            compute_metrics=compute_metrics_huggingface,
        )
        trainer.train()

        # set pipe for inference
        self.inference_pipe = TextClassificationPipeline(
            model=self.engine,
            tokenizer=self.features_tokenizer.tokenizer,
            return_all_scores=True,
        )

    def predict(self, dataset: Union[pd.DataFrame, Dataset]) -> pd.DataFrame:
        """Make predictions.

        Args:
            dataset: data with feature_key.

        Returns:
            The prediction score for each class, plus two columns with the higest score
            and the corresponding class (predicted_scores and predicted_labels).
        """
        # format input
        if isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset)
        input_data = dataset[self.features_tokenizer.feature_key]

        # make predictions
        preds = self.inference_pipe(input_data)

        # format output
        preds = pd.concat(
            [pd.DataFrame(d).set_index("label").transpose() for d in preds]
        ).reset_index(drop=True)
        preds.columns.name = None
        preds = preds.rename(
            columns={
                f"LABEL_{i}": v
                for i, v in self.labels_tokenizer.labels_dict_inverse.items()
            }
        )
        predicted_label = preds.idxmax(axis=1)
        predicted_score = preds.max(axis=1)
        preds["predicted_labels"] = predicted_label
        preds["predicted_scores"] = predicted_score

        return preds

    def load_data(self, data="train"):
        """Load data from a folder.

        Args:
            data: Must be one of "train","valid","test".

        Returns:
            The loaded data.
        """
        assert data in {"train", "valid", "test"}
        if self.path_run_dir is None:
            last_run = sorted(
                [v for v in os.listdir(self.path_data_dir) if "run_" in v]
            )[-1]
            self.path_run_dir = f"{self.path_data_dir}/{last_run}/"

        return pd.read_pickle(f"{self.path_run_dir}/{data}.p")


class FeaturesTokenizer:
    """Tokenizer for features."""

    def __init__(
        self,
        checkpoint,
        cache_dir,
        feature_key,
        truncation=True,
        max_length=500,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.feature_key = feature_key
        self.truncation = truncation
        self.max_length = max_length

    def __call__(self, x):
        return self.tokenizer(
            x[self.feature_key],
            truncation=self.truncation,
            max_length=self.max_length,
        )


class LabelsTokenizer:
    """Tokenizer for labels.

    This class is responsable to convert raw_labels to integers (and viceversa).
    """

    def __init__(self, labels):
        labels = sorted(set(labels))
        self.labels_dict_inverse = {idx: value for idx, value in enumerate(labels)}
        self.labels_dict = {
            value: idx for idx, value in self.labels_dict_inverse.items()
        }
        self.num_labels = len(labels)

    def __call__(self, x):
        x["labels"] = [self.labels_dict[label] for label in x["raw_labels"]]
        return x
