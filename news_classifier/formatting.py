import os
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


def get_formatter(formatter_class="StandardFormatter", formatter_params={}):
    return getattr(sys.modules[__name__], formatter_class)(**formatter_params)


@dataclass
class StandardFormatter:
    """
    If path_raw_file is None, take the last run in path_raw_dir.
    """

    path_raw_file: str = None
    path_raw_dir: str = "data/raw/"
    seed: int = 123

    def run(self):
        """
        Load data, convert to dataframe, add item_id, split in train/valid/test.

        Raw data is expected to be a dict mapping categories to a list of dict with
        different properties.
        """
        if not self.path_raw_file:
            last_run = sorted(
                [v for v in os.listdir(self.path_raw_dir) if "run_" in v]
            )[-1]
            self.path_raw_file = f"{self.path_raw_dir}/{last_run}/data.p"

        # load
        data = pd.read_pickle(self.path_raw_file)
        # to dataframe
        data = _convert_to_dataframe(data)
        # assign item_id column
        data = data.reset_index(drop=True).reset_index().rename(columns={"index":"item_id"})
        # split
        train, valid, test = _split_data(data, self.seed)
        return train, valid, test


def _split_data(data, seed=123):
    """
    Split data in train (50%), valid (25%), and test (25%).
    Stratify by "labels".
    """
    # split in train valid test
    test_valid_size = len(data) // 4
    tmp, test = train_test_split(
        data, test_size=test_valid_size, stratify=data["labels"], random_state=seed
    )
    train, valid = train_test_split(
        tmp, test_size=test_valid_size, stratify=tmp["labels"], random_state=seed
    )
    return train, valid, test


def _convert_to_dataframe(dataset):
    df = []
    for category, l in dataset.items():
        tmp = pd.DataFrame(l)
        tmp["labels"] = category
        df.append(tmp)
    return pd.concat(df)
