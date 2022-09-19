"""
@Author  :   Sander Schwenk-Nebbe
@Contact :   sandersn@econ.au.dk
@License :   MIT
"""

from __future__ import annotations

from typing import Callable, Union

import pandas as pd
import numpy as np


class AnatomyModel:
    """
    Contains a model's prediction function
    """
    def __init__(self, pred_fn: Callable[[np.ndarray], np.ndarray]):
        """
        :param pred_fn: prediction function of model (takes 2-dim X returns 1-dim y_hat)
        """
        self.predict = pred_fn


class AnatomyModelProvider:
    """
    Provides train set, test set, and model for a given prediction period and model name (e.g.,
     by loading it from disk or by training it ad hoc)
    """
    class PeriodKey:
        def __init__(self, period: int, model_name: str):
            self.period = period
            self.model_name = model_name

    class PeriodValue:
        def __init__(self, train: pd.DataFrame, test: pd.DataFrame, model: AnatomyModel):
            self.train = train
            self.test = test
            self.model = model

    def __init__(
        self,
        n_periods: int,
        n_features: int,
        model_names: list[str],
        y_name: str,
        provider_fn: Callable[[PeriodKey], PeriodValue]
    ):
        """
        :param n_periods: number of predictions (number of test sets)
        :param n_features: number of features (predictors)
        :param model_names: list of model names
        :param y_name: name of the target (y) in the train and test sets
        :param provider_fn: function that returns train, test and model for a prediction period (e.g., by training
         model or loading it from disk if already trained)
        """
        self.n_periods = n_periods
        self.n_features = n_features
        self.model_names = model_names
        self.y_name = y_name
        self.provider_fn = provider_fn


class AnatomyModelCombination:
    def __init__(self, groups: Union[dict[str, list[str]], dict[str, dict[str, float]]]):
        """
        :param groups: dict of combination name and list of model names or dict of model names and their weights
         in the combination
        """
        self.groups = groups


class AnatomyModelOutputTransformer:
    def __init__(
        self,
        transform: Union[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]
    ):
        """
        :param transform: function used to transform model output; must have named argument y_hat (contains
         raw model output) and can optionally use named argumeny y (contains true target);
         let transformer return aggregated result instead of series to get an aggregated (global) explanation
        """
        assert "y_hat" in transform.__code__.co_varnames
        assert all([x in ["y_hat", "y"] for x in transform.__code__.co_varnames])
        self.transform = transform
