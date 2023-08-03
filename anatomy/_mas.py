"""
@Author  :   Sander Schwenk-Nebbe
@Contact :   sandersn@econ.au.dk
@License :   MIT
"""

from __future__ import annotations

import math

from enum import Enum

import numpy as np
import pandas as pd


class MAS:

    @staticmethod
    def _s(n):
        """
        Sum of squares of n natural numbers.
        """
        return n * (n+1) * (2*n+1) / 6

    @staticmethod
    def _c(n, k):
        """
        Binomial coefficient.
        """
        return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

    @staticmethod
    def _expected_msdr(
        p,
        alpha=0.5
    ) -> float:
        """
        Expected mean squared deviation in ranks of A=rank(*) and B=signed-rank(*) of p elements.
        """
        assert alpha == 0.5
        return 1 / p * (
            MAS._s(p) + sum([MAS._c(p, a) * (0.5 ** p) * (MAS._s(a) + MAS._s(p - a)) for a in range(0, p + 1)])
        )

    class LossType(Enum):
        LOWER_IS_BETTER = 1
        LARGER_IS_BETTER = 2

    class H0:
        """
        Computes the distribution of Model Accordance Scores under the null of unrelated ranks with good predictors
        as likely as bad predictors (alpha=0.5).
        """
        def __init__(self, p, n_samples=1000000, alpha=0.5):
            self.p = p
            self.alpha = 0.5
            self.mas_h0: np.ndarray = (
                1 - self._generate_msdr_h0(p=p, n_samples=n_samples, alpha=alpha) / MAS._expected_msdr(p)
            )

        @staticmethod
        def _generate_msdr_h0(
            p,
            n_samples,
            alpha
        ) -> np.ndarray:
            """
            Generates MSDR under null hypothesis of unrelated ranks A and signed-ranks B, with number of
            positive (good) ranks in B picked from binomial(p, alpha). With alpha=0.5, positive (good) ranks are as
            likely as negative (bad) ranks.
            """
            A = np.arange(1, p + 1)
            msdr = np.empty(n_samples)
            p_good = np.random.binomial(p, alpha, n_samples)
            shuffle_idx_a = np.argsort(np.random.uniform(-1, 1, (n_samples, p)), axis=1)
            shuffle_idx_b = np.argsort(np.random.uniform(-1, 1, (n_samples, p)), axis=1)
            for i in range(n_samples):
                n_good = p_good[i]
                n_bad = p - n_good
                B = np.hstack((A[:n_good], -A[:n_bad]))[shuffle_idx_b[i]]
                msdr[i] = np.mean((A[shuffle_idx_a[i]] - B) ** 2)
            return msdr

    def __init__(self, variable_importance: pd.Series, pbsv: pd.Series, pbsv_loss_type: LossType, h0: H0 | None = None):

        assert variable_importance.shape == pbsv.shape
        assert all(variable_importance.index == pbsv.index)
        assert "base_contribution" not in variable_importance.index

        self._p = variable_importance.shape[0]
        if h0 is not None:
            assert h0.p == self._p

        self._vi = variable_importance
        self._pbsv = pbsv
        self._loss_type = pbsv_loss_type
        self._h0 = h0

    def compute(self, h0: H0 | None = None):
        if h0 is not None:
            assert h0.p == self._p
        _h0 = h0 if h0 is not None else self._h0

        vi_ranked = MAS._rank(self._vi)
        pbsv_signed_ranked = MAS._signed_rank(self._pbsv, self._loss_type)

        msdr = np.mean((vi_ranked - pbsv_signed_ranked) ** 2)

        mas = 1 - msdr / MAS._expected_msdr(self._p)
        mas_p_val = None

        if _h0 is not None:
            mas_p_val = (_h0.mas_h0 > mas).mean()

        return {
            "mas": mas,
            "mas_p_val": mas_p_val
        }

    @staticmethod
    def _signed_rank(pbsv, loss_type: LossType) -> pd.Series:

        if loss_type == MAS.LossType.LARGER_IS_BETTER:
            signed_ranks = pd.concat((
                pbsv[pbsv >= 0].rank(ascending=True),
                -pbsv[pbsv < 0].rank(ascending=False)
            ))
        else:
            signed_ranks = pd.concat((
                pbsv[pbsv <= 0].rank(ascending=False),
                -pbsv[pbsv > 0].rank(ascending=True)
            ))
        signed_ranks = signed_ranks.reindex(pbsv.index)

        return signed_ranks

    @staticmethod
    def _rank(vi: pd.Series) -> pd.Series:
        return vi.rank(ascending=True)
