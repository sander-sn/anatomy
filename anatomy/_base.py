"""
@Author  :   Sander Schwenk-Nebbe
@Contact :   sandersn@econ.au.dk
@License :   MIT
"""

from __future__ import annotations

from typing import Union
from enum import Enum
import warnings
import pickle

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend

from . import AnatomyModelProvider, AnatomyModelCombination, AnatomyAlgorithm, AnatomyModelOutputTransformer

class Anatomy:

    class ExplanationLevel(Enum):
        LOCAL = 1
        GLOBAL = 2

    def __init__(
        self,
        provider: Union[AnatomyModelProvider, None],
        n_iterations=1
    ):
        """
        :param provider: provides train set, test set, and model for given prediction period and model name
        :param n_iterations: number of permuted orderings the models are evaluated on
        """
        self._provider = provider
        self._n_periods = None if provider is None else provider.n_periods
        self._n_features = None if provider is None else provider.n_features
        self._y_name = None if provider is None else provider.y_name
        self._model_names = None if provider is None else provider.model_names
        self._permutations = None if provider is None else self._gen_permutations(self._n_features, n_iterations)

        self._columns = None
        self._xy_train = None
        self._xy_test = None
        self._y_hats = None
        self._Y = None

    def get_forecast_index(self) -> Union[pd.Index, None]:
        """
        :return: index of entire series of forecasts (the index of concatenated test sets)
        """
        if self._xy_test is None:
            return None
        return self._xy_test.index

    @staticmethod
    def _df_to_rec_dict(df):
        return {
            "index": {"vals": df.index.to_numpy(), "name": df.index.name},
            "data": {"vals": df.to_records(index=False), "columns": df.columns}
        }

    @staticmethod
    def _rec_dict_to_df(rec_dict):
        df = pd.DataFrame(
            data=rec_dict["data"]["vals"],
            index=pd.Index(rec_dict["index"]["vals"], name=rec_dict["index"]["name"]),
        )
        df.columns = rec_dict["data"]["columns"]
        return df

    def save(self, path: str) -> None:
        """
        Saves a precomputed Anatomy to disk
        """

        vars_to_save = [
            "_n_periods", "_n_features", "_y_name", "_model_names", "_permutations", "_columns", "_y_hats", "_Y"
        ]

        vars_to_rec_dict = [
            "_xy_train", "_xy_test"
        ]

        content = {key: getattr(self, key) for key in vars_to_save}
        content = dict(content, **{key: self._df_to_rec_dict(getattr(self, key)) for key in vars_to_rec_dict})

        pickle.dump(content, open(path, "wb"))

    @staticmethod
    def load(path: str) -> Anatomy:
        """
        Loads an already precomputed Anatomy from disk
        """

        vars_from_rec_dict = [
            "_xy_train", "_xy_test"
        ]

        content: dict = pickle.load(open(path, "rb"))
        obj = Anatomy(provider=None)
        for key, value in content.items():
            setattr(obj, key, Anatomy._rec_dict_to_df(value) if key in vars_from_rec_dict else value)
        return obj

    @staticmethod
    def _gen_permutations(n_features, n_iterations) -> np.ndarray:
        return np.array([np.random.permutation(n_features) for _ in range(n_iterations)]).T

    @staticmethod
    def _run_job(
        provider: AnatomyModelProvider,
        permutations: np.ndarray,
        model_name: str,
        t: int,
        subsample: float
    ) -> tuple[tuple[str, int], dict]:
        payload_key = (model_name, t)

        provider_key = AnatomyModelProvider.PeriodKey(period=t, model_name=model_name)
        provider_value = provider.provider_fn(provider_key)

        model = provider_value.model
        train_full, test_full = provider_value.train, provider_value.test

        train = train_full.drop(provider.y_name, axis=1)
        test = test_full.drop(provider.y_name, axis=1)

        y_hat = model.predict(test.to_numpy())

        Y = AnatomyAlgorithm.evaluate_permuted_orders(
            model=model,
            X=train,
            xs=test,
            permutations=permutations,
            subsample=subsample
        )

        res = {
            "Y": Y,
            "train": train_full,
            "test": test_full,
            "y_hat": y_hat
        }

        return payload_key, res

    def precompute(
        self,
        n_jobs: int = 5,
        background_data_subsample: float = 1,
        save_path: Union[str, None] = None
    ) -> Anatomy:
        """
        Precomputes model outputs in permuted orderings; after this step all models, periods, and transformations
        can be explained at no additional computational time

        :param n_jobs: number of parallel estimation processes (max one per prediction period and model)
        :param background_data_subsample: fraction of training data to use to replace deactivated (masked) features
        :param save_path: path (including file name) to save the precomputed results to
        """

        assert self._provider is not None
        assert self._Y is None

        if save_path is not None:
            with open(save_path, "a") as f:
                assert f.writable()

        payloads = []

        for model_name in self._model_names:

            for t in range(self._n_periods):

                payloads.append(delayed(self._run_job)(
                    provider=self._provider,
                    permutations=self._permutations,
                    model_name=model_name,
                    t=t,
                    subsample=background_data_subsample
                ))

        with parallel_backend(backend="loky", inner_max_num_threads=1):  # force one thread (e.g., OMP_NUM_THREADS=1)
            p = Parallel(n_jobs=n_jobs, verbose=max(50+1, len(payloads)))
            results_dict = dict(p(payloads))

        self._Y = np.stack([
            np.vstack([
                results_dict[model_name, t]["Y"]
                for t in range(self._n_periods)
            ])
            for model_name in self._model_names
        ])

        self._y_hats = np.stack([
            np.hstack([
                results_dict[model_name, t]["y_hat"]
                for t in range(self._n_periods)
            ])
            for model_name in self._model_names
        ])

        xy_test = [
            pd.concat([
                results_dict[model_name, t]["test"]
                for t in range(self._n_periods)
            ])
            for model_name in self._model_names
        ]
        assert all([xy_test[0].equals(x) for x in xy_test])
        self._xy_test = xy_test[0]

        xy_train = [
            pd.concat([
                results_dict[model_name, t]["train"]
                for t in range(self._n_periods)
            ])
            for model_name in self._model_names
        ]
        assert all([xy_train[0].equals(x) for x in xy_train])
        self._xy_train = xy_train[0]

        self._columns = xy_train[0].drop(self._y_name, axis=1).columns.to_list()

        self._check_efficiency()

        if save_path is not None:
            self.save(path=save_path)

        return self

    def _check_efficiency(self):
        recovered_y_hats = self.explain().sum(axis=1)
        for i, model_name in enumerate(self._model_names):
            if not np.isclose(self._y_hats[i, :], recovered_y_hats.xs(model_name).to_numpy()).all():
                warnings.warn(
                    "efficiency check failed; "
                    "Shapley values for model %s do not sum to predictions" % model_name
                )

    def explain(
        self,
        model_sets: Union[AnatomyModelCombination, None] = None,
        transformer: Union[AnatomyModelOutputTransformer, None] = None,
        explanation_subset: Union[pd.Index, None] = None
    ) -> pd.DataFrame:
        """
        Explains the forecasts or subset of forecasts, model or combination of models, and forecasts or transformations
        of forecasts (e.g., loss or gain conditioned on forecast)

        :param model_sets: combination of models
        :param transformer: transform the forecasts (e.g., apply loss or gain function)
        :param explanation_subset: subset
        :return: dataframe of base contributions (conditioned on empty coalition) and contributions allocated across
         all features; summing across base and feature contributions (columns in the dataframe) yields the value that
         is explained (forecasts or transformed forecasts) by the efficiency property of Shapley values
        """
        assert self._Y is not None

        if model_sets is None:
            model_sets = AnatomyModelCombination(groups={x: [x] for x in self._model_names})

        if transformer is None:
            transformer = AnatomyModelOutputTransformer(transform=lambda y_hat: y_hat)

        if explanation_subset is None:
            explanation_subset = self._xy_test.index
        subset = self._xy_test.index.isin(explanation_subset)
        
        assert subset.any()

        def _apply(comb_set: Union[list[str], dict[str, float]]):

            models = comb_set if type(comb_set == list) else comb_set.keys()
            weights = np.repeat(1.0, len(comb_set)) if type(comb_set == list) else np.array(comb_set.values())

            model_indices = [self._model_names.index(x) for x in models]

            if "y" in transformer.transform.__code__.co_varnames:
                transform = lambda y_hat: transformer.transform(y_hat=y_hat, y=self._xy_test[subset][self._y_name].to_numpy())
            else:
                transform = lambda y_hat: transformer.transform(y_hat=y_hat)

            Y_combination = np.average(self._Y[model_indices][:, subset, :, :, :], weights=weights, axis=0)
            Y = np.apply_along_axis(transform, axis=0, arr=Y_combination)

            assert len(Y.shape) in [3, 4]  # 3 => AGGREGATED (GLOBAL) ; 4 => LOCAL
            explanation_level = Anatomy.ExplanationLevel.LOCAL if len(Y.shape) == 4 else Anatomy.ExplanationLevel.GLOBAL

            if explanation_level == Anatomy.ExplanationLevel.GLOBAL:
                Y = Y.reshape(1, *Y.shape)

            base_contribution = Y[:, 0, 0, 0]
            marginal_contributions = np.diff(Y, axis=1)

            Phi = np.zeros_like(marginal_contributions)
            for m in range(Phi.shape[-2]):
                Phi[:, self._permutations[:, m], m, 0] = marginal_contributions[:, :, m, 0]
                Phi[:, self._permutations[:, m][::-1], m, 1] = marginal_contributions[:, :, m, 1]
            Phi = Phi.mean(axis=-1).mean(axis=-1)

            explanation = pd.DataFrame(
                np.hstack((base_contribution.reshape(-1, 1), Phi)),
                columns=["base_contribution"] + self._columns
            )

            if explanation_level == Anatomy.ExplanationLevel.GLOBAL:
                explanation.index = [
                    "%s -> %s" % (str(self._xy_test[subset].index[0]), str(self._xy_test[subset].index[-1]))
                ]
            elif explanation_level == Anatomy.ExplanationLevel.LOCAL:
                explanation.index = self._xy_test[subset].index

            return explanation

        df = pd.concat([_apply(x) for x in model_sets.groups.values()], keys=model_sets.groups.keys())
        return df
