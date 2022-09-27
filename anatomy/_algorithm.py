"""
@Author  :   Sander Schwenk-Nebbe
@Contact :   sandersn@econ.au.dk
@License :   MIT
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from . import AnatomyModel


class AnatomyAlgorithm:

    @staticmethod
    def evaluate_permuted_orders(
        model: AnatomyModel,
        X: pd.DataFrame,
        xs: pd.DataFrame,
        permutations: np.ndarray,
        subsample: float = 1.0
    ) -> np.ndarray:
        """
        Activates (unmasks) features in permuted order and records model output.

        Features are deactivated (masked) by replacing them with data from the training dataset (or a subsample hereof
        when param subsamlpe < 1.0) and are then activated one by one in the permuted orderings provided, each time
        recording the raw model output.

        :param model: model with prediction function to evaluate
        :param X: training set with which model was trained (without target)
        :param xs: test set to evaluate prediction function on (without target)
        :param permutations: array of permuted orderings of n_features
        :param subsample: subsample of training set to replace deactivated (masked) features with
        :return: array of model outputs with masked features in permuted orderings with shape [U, J + 1, M, 2] where
         U is obs in xs, J is n_features (at J=0, model is evaluated with no active features),
         M is number of permuted orderings to evaluate, and last dimension contains the
         original and reversed permutation (antithetic sampling)
        """

        f = model.predict

        assert 0 < subsample <= 1
        X = (X if subsample == 1 else X.sample(frac=subsample)).to_numpy(copy=True)
        assert X.shape[0] > 0

        xs = xs.to_numpy(copy=True)

        # number of obs T and features J in training set X, number of obs U in test set xs, number of permutations M:
        (T, J), U, M = X.shape, xs.shape[0], permutations.shape[1]

        # allocate matrix to hold raw model outputs evaluated in permuted orderings (original and reverse):
        Y = np.zeros((U, J + 1, M, 2), dtype=np.float64)

        for u in range(U):

            x = xs[u, :]
            J_ms = permutations.copy()

            # runs with none and all features activated are invariant to ordering;
            # calculate model output for the two masks outside loop:
            y_hat_none_all = f(np.vstack((X, x))).flatten()
            y_hat_avg_none, y_hat_avg_all = y_hat_none_all[:T].mean(), y_hat_none_all[T]

            for m in range(M):

                # obtain order in which features are activated (permutation of J):
                J_m = J_ms[:, m]

                for forwards_backwards in [0, 1]:

                    # reverse permutation for antithetic sampling:
                    if forwards_backwards == 1:
                        J_m = J_m[::-1]

                    # obtain copy of original training data which is modified in this run:
                    X_m = X.copy()

                    # allocate matrix to hold all feature activations (excluding none and all active features)
                    # to evaluate model once potentially reducing overhead in calling f:
                    X_eval_m = np.zeros((T * (J - 1), J), dtype=np.float64)

                    # activate feature i in permutation J_m excluding last feature at
                    # which all features would be activated (which we already have):
                    for i in range(J - 1):
                        j = J_m[i]
                        # replace all values of column j in X_m with value of column j in
                        # row to explain (x) thereby activating feature j in X_m:
                        X_m[:, j] = x[j]
                        # place current X_m in aggregate evaluation matrix:
                        X_eval_m[T * i:T * (i + 1)] = X_m

                    # evaluate J-1 activations;
                    # each activation is evaluated on T obs;
                    # thus reshape output to (J-1)xT matrix;
                    # take average over all T;
                    # this yields the average model output for each activation:
                    y_hat_avg_m = f(X_eval_m).flatten().reshape(J - 1, -1).mean(axis=-1)

                    # add first (no features active) and last (all features active) masking:
                    y_hat_avg_m = np.hstack((y_hat_avg_none, y_hat_avg_m, y_hat_avg_all))

                    # store results:
                    Y[u, :, m, forwards_backwards] = y_hat_avg_m

        return Y
