"""
@Author  :   Sander Schwenk-Nebbe
@Contact :   sandersn@econ.au.dk
@License :   MIT
"""

from __future__ import annotations

from enum import Enum

import pandas as pd


class AnatomySubset:
    def __init__(self, train_subset: slice, test_subset: slice):
        assert test_subset.start >= train_subset.stop
        self.train_subset = train_subset
        self.test_subset = test_subset


class AnatomySubsets:

    class EstimationType(Enum):
        EXPANDING = 1
        ROLLING = 2

    @staticmethod
    def generate(
        index: pd.Index,
        initial_window: int,
        estimation_type: EstimationType,
        periods: int = 1,
        gap: int = 0
    ) -> AnatomySubsets:
        """
        Generates train and test set subsets of a dataset for a forecasting application

        :param index: index of entire dataset that is split in train and test sets
        :param initial_window: obs of first train set
        :param estimation_type: whether train set expands or rolls forward
        :param periods: obs to predict each time (size of test sets)
        :param gap: gap between train and test sets
        :return: subsets of train and test sets
        """

        assert periods > 0
        assert gap >= 0
        assert 0 < initial_window < index.shape[0] - 1
        assert index.is_monotonic_increasing and index.is_unique

        t, subsets = 0, []

        while True:

            if t + initial_window + gap + 1 > index.shape[0]:
                break

            if estimation_type == AnatomySubsets.EstimationType.ROLLING:
                train = slice(t, t+initial_window)
            else:
                train = slice(0, t+initial_window)
            test = slice(t + initial_window + gap, min(t + initial_window + gap + periods, index.shape[0]))

            subsets.append(AnatomySubset(train_subset=train, test_subset=test))

            t += periods

        return AnatomySubsets(index=index, subsets=subsets)

    def __init__(self, index: pd.Index, subsets: list[AnatomySubset]):
        self._index = index
        self._subsets = subsets
        self.n_periods = len(subsets)

    def get_train_subset(self, period: int) -> slice:
        assert 0 <= period < self.n_periods
        return self._subsets[period].train_subset

    def get_test_subset(self, period: int) -> slice:
        assert 0 <= period < self.n_periods
        return self._subsets[period].test_subset

    def check_index_eq(self, index: pd.Index):
        eq = index.to_numpy() == self._index.to_numpy()
        return eq if type(eq) == bool else eq.all()
