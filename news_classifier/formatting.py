import os
import sys
import pandas as pd
from typing import Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


def get_formatter(
    formatter_class: str = "StandardFormatter",
    formatter_params: Union[Dict, None] = None,
) -> Any:
    """Factory for formatter objects.

    Args:
        formatter_class: The name of the formatter class.
        formatter_params: The params for the formatter class.

    Returns:
        Formatter object.
    """
    if formatter_params is None:
        formatter_params = {}
    return getattr(sys.modules[__name__], formatter_class)(**formatter_params)


@dataclass
class StandardFormatter:
    """Format the data.

    The following actions are done:
    - load data
    - merge categories
    - merge columns
    - split data in train/valid/test
    - sample training set

    Check the args for details.

    Args:
        path_raw_file: Path to the raw data pickle file. Raw data is expected to be a dict
            mapping categories to a list of dict with different properties.
        path_raw_dir: Path to the raw data dir. Ignored if path_raw_file is not None.
            If path_raw_file is None, take the last run in path_raw_dir.
        sample_how_many: How many samples to take per class. Do not perform sampling if
            set to None.
        sample_replace: If True, sample with replacement.
        merge_columns: List of string columns to be "merged" to create a new column.
            Ignored if None.
        merge_categories: Dict mapping macro-categories to list of categories. Ignored
            if None.
        merge_sep: separator to use to merge columns.
        seed: random seed to split train/valid/test sets.
    """

    path_raw_file: Union[str, None] = None
    path_raw_dir: Union[str, None] = "data/raw/"
    sample_how_many: Union[int, None] = None
    sample_replace: bool = False
    merge_columns: Union[List[str], None] = None
    merge_categories: Union[Dict[str, List[str]], None] = None
    merge_sep: str = "\n"
    seed: int = 123

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the data formatting.

        Returns:
            The data formatted as train/valid/test
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
        # merge categories
        if self.merge_categories is not None:
            data = _merge_categories(data, self.merge_categories)
        # assign item_id column
        data = (
            data.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "item_id"})
        )
        # merge columns
        if self.merge_columns is not None:
            data = _merge_columns(data, self.merge_columns, self.merge_sep)
        # split
        train, valid, test = _split_data(data, self.seed)
        # sample
        train = sample_data(
            train, how_many=self.sample_how_many, replace=self.sample_replace
        )
        return train, valid, test


def _split_data(
    data: pd.DataFrame, seed: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data in train (50%), valid (25%), and test (25%), stratify by "raw_labels".

    Args:
        data: The dataset.
        seed: Random seed used for split.

    Returns:
        The train/valid/test sets.
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


def sample_data(
    data: pd.DataFrame, how_many: Union[int, None] = None, replace: bool = True
) -> pd.DataFrame:
    """Resample data based on raw_labels.

    Args:
        data: Dataframe with `raw_labels` column.
        how_many: How many samples per class to take. If None, return unchanged data.
        replace: Sample with/without replacement. Ignored if how_many is None. If
            False, take the min of the total number of samples and how_many.

    Returns:
        A new dataframe with the resampled rows.
    """
    if how_many is None:
        return data.copy()

    if replace:
        sampler = lambda x: x.sample(n=how_many, replace=True)
    else:
        sampler = lambda x: x.sample(n=min(how_many, len(x)))

    return data.groupby("raw_labels").apply(sampler).reset_index(drop=True)


def _convert_to_dataframe(dataset: Dict) -> pd.DataFrame:
    """Convert data dict to dataframe."""
    df = []
    for raw_label, l in dataset.items():
        tmp = pd.DataFrame(l)
        tmp["raw_labels"] = raw_label
        df.append(tmp)
    return pd.concat(df)


def _merge_columns(
    data: pd.DataFrame, merge_columns: List[str], merge_sep: str
) -> pd.DataFrame:
    """Create new column in dataframe by merging existing string columns."""
    data = data.copy()
    tmp = data[merge_columns[0]].copy()
    for k in merge_columns[1:]:
        tmp = tmp + merge_sep + data[k]
    new_col = "_".join(merge_columns)
    data[new_col] = tmp
    return data


def _merge_categories(
    data: pd.DataFrame, merge_categories: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Merge categories in `raw_labels` according to merge_categories.
    Store old categories in a new column `old_raw_labels`.
    """
    data = data.copy()
    data["old_raw_labels"] = data["raw_labels"].copy()

    for macro_cat, categories in merge_categories.items():
        mask = data.raw_labels.isin(categories)
        data.loc[mask, "raw_labels"] = macro_cat

    return data
