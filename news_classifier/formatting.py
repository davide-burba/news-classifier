import os
import sys
import pandas as pd
from typing import Union, List
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
    sample_how_many: str = None
    sample_replace: bool = False
    merge_columns: Union[List[str], None] = None
    merge_sep: str = "\n"
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

        # load
        data = pd.read_pickle(self.path_raw_file)
        # to dataframe
        data = _convert_to_dataframe(data)
        # assign item_id column
        data = (
            data.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "item_id"})
        )
        # merge columns
        if self.merge_columns:
            data = _merge_columns(data, self.merge_columns, self.merge_sep)
        # split
        train, valid, test = _split_data(data, self.seed)
        # sample
        train = sample_data(
            train, how_many=self.sample_how_many, replace=self.sample_replace
        )
        return train, valid, test


def _split_data(data, seed=123):
    """
    Split data in train (50%), valid (25%), and test (25%).
    Stratify by "raw_labels".
    """
    # split in train valid test
    test_valid_size = len(data) // 4
    tmp, test = train_test_split(
        data, test_size=test_valid_size, stratify=data["raw_labels"], random_state=seed
    )
    train, valid = train_test_split(
        tmp, test_size=test_valid_size, stratify=tmp["raw_labels"], random_state=seed
    )
    return train, valid, test


def _convert_to_dataframe(dataset):
    df = []
    for raw_label, l in dataset.items():
        tmp = pd.DataFrame(l)
        tmp["raw_labels"] = raw_label
        df.append(tmp)
    return pd.concat(df)


def sample_data(data, how_many=None, replace=True):
    """
    Resample data based on raw_labels.

    Args:
        data (pd.DataFrame): Dataframe with raw_labels column
        how_many (Union[int,None]): How many samples per class to take. If None, return
            unchanged data.
        replace (bool): Sample with/without replacement. Ignored if how_many is None. If
            False, take the min of the total number of samples and how_many.

    Returns:
        pd.DataFrame: A new dataframe with the resampled rows
    """
    if how_many is None:
        return data.copy()

    if replace:
        sampler = lambda x: x.sample(n=how_many, replace=True)
    else:
        sampler = lambda x: x.sample(n=min(how_many, len(x)))

    return data.groupby("raw_labels").apply(sampler).reset_index(drop=True)


def _merge_columns(data, merge_columns, merge_sep):
    data = data.copy()
    tmp = data[merge_columns[0]].copy()
    for k in merge_columns[1:]:
        tmp = tmp + merge_sep + data[k]
    new_col = "_".join(merge_columns)
    data[new_col] = tmp
    return data
