"""
Operations with recent files list.

This file contains the following functions:
    * read_preferences
"""

import pickle
from pathlib import Path

from ..data.collections import IndexedDeque
from ..data.config import get_config


class RecentList(IndexedDeque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = get_config()["recent"]["path"]
        if not self.exists():
            self.save()
        else:
            self.read()

    def __len__(self):
        """
        Override method.
        """
        return self.deque.__len__()

    def __iter__(self):
        """
        Override method.
        """
        return self.deque.__iter__()

    def exists(self) -> bool:
        """
        Check recent_list exists.

        Returns
        -------
        out: bool
            exists or not
        """
        return Path(self.path).exists()

    def save(self) -> None:
        """
        Save deque and set using pickle
        """
        with open(self.path, "wb") as f:
            pickle.dump(vars(self), f, protocol=pickle.HIGHEST_PROTOCOL)

    def read(self) -> None:
        """
        Read deque and set from pickle file.
        """
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
