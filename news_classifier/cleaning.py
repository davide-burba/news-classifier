import os
import pandas as pd
from typing import Union, Dict, List
from dataclasses import dataclass


@dataclass
class Cleaner:
    """Class responsible for cleaning the data."""

    path_raw_dir: Union[str, None] = None
    path_raw_file: Union[str, None] = None
    words_per_abstract: Union[int, None] = 10

    def run(self) -> Dict[str, List[Dict]]:
        """Run the data cleaning.

        The following operations are done:
        - keep only articles with abstract length > 10

        """
        data = self._load_data()

        # keep only articles with abstract length > 10
        if self.words_per_abstract is not None:
            data = {
                k: [
                    d for d in l if len(d["abstract"].split()) > self.words_per_abstract
                ]
                for k, l in data.items()
            }

        return data

    def _load_data(self):
        if not self.path_raw_file:
            last_run = sorted(
                [v for v in os.listdir(self.path_raw_dir) if "run_" in v]
            )[-1]
            self.path_raw_file = f"{self.path_raw_dir}/{last_run}/data.p"

        return pd.read_pickle(self.path_raw_file)
