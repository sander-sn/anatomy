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
        p: int,
        alpha=0.5
    ) -> float:
        """
        Expected value of (A-B)^2 where A=rank(a) and B=signed-rank(b) and a and b have 'p' elements and 'a' is
        independent of 'b' and a negative is as likely as a positive element in 'b' (because 'alpha = 0.5').
        :param p: Number of ranks (elements in a and b).
        """
        assert p > 1
        assert alpha == 0.5
        return 1 / p * (
            MAS._s(p) + sum([MAS._c(p, a) * (0.5 ** p) * (MAS._s(a) + MAS._s(p - a)) for a in range(0, p + 1)])
        )

    @staticmethod
    def _expected_msdr_w(
        w: np.ndarray,
        A: np.npdarray,
        alpha=0.5
    ) -> float:
        """
        Expected value of w*(A-B)^2 where a>0, w=a/mean(a), A=rank(a), and B=signed-rank(b) and 'a' and 'b' have
        'p' elements and 'a' is independent of 'b' and a negative is as likely as a positive element in 'b'
        (because 'alpha = 0.5').
        :param w: Vector of mean-one weights, 'w = a / mean(a)'.
        :param A: Vector of ranks 'A = rank(a)'.
        """
        assert alpha == 0.5
        assert len(A) == len(w)

        p = len(A)

        return 1 / p * (
            sum(w*(A**2)) + sum([MAS._c(p, a) * (0.5 ** p) * (MAS._s(a) + MAS._s(p - a)) for a in range(0, p + 1)])
        )

    class LossType(Enum):
        LOWER_IS_BETTER = 1
        LARGER_IS_BETTER = 2

    class MASType(Enum):
        IMPORTANCE_WEIGHTED = 1
        EQUAL_WEIGHTED = 2

    class H0:
        """
        Computes the distribution of Model Accordance Scores under the null of unrelated ranks.
        """
        def __init__(self, p: int, mas_type: MAS.MASType, n_samples: int = 1000000, alpha: float = 0.5,
                     is_vi_w: None | np.ndarray = None, is_vi_r: None | np.ndarray = None):
            """
            :param alpha: The porportion of good predictors under the null. At 'alpha = 0.5', good predictors
            are as likely as bad predictors.
            """
            assert 0 <= alpha <= 1
            assert mas_type == MAS.MASType.EQUAL_WEIGHTED or (
                mas_type == MAS.MASType.IMPORTANCE_WEIGHTED and
                is_vi_w is not None and
                is_vi_r is not None
            )
            self.p = p
            self.alpha = 0.5
            self.mas_type = mas_type
            self.w = is_vi_w
            self.A = is_vi_r

            if self.mas_type == MAS.MASType.EQUAL_WEIGHTED:
                self.msdr_h0 = (
                    self._generate_msdr_h0(p=p, n_samples=n_samples, alpha=alpha)
                )
            elif self.mas_type == MAS.MASType.IMPORTANCE_WEIGHTED:
                self.msdr_h0_w = (
                    self._generate_msdr_h0_w(p=p, n_samples=n_samples, alpha=alpha, w=self.w, A=self.A)
                )

        def get_p_value(self, observed_msdr: float):
            """
            :param observed_msdr: The (weighted) observed.
            :return: Yields the empirical probability of observing a (weighted) mean squared deviation in ranks less
            than or as extreme as the 'observed_msdr' under the null.
            """
            if self.mas_type == MAS.MASType.EQUAL_WEIGHTED:
                return (self.msdr_h0 <= observed_msdr).mean()
            elif self.mas_type == MAS.MASType.IMPORTANCE_WEIGHTED:
                return (self.msdr_h0_w <= observed_msdr).mean()

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

        @staticmethod
        def _generate_msdr_h0_w(
            p,
            n_samples,
            alpha,
            w,
            A
        ) -> np.ndarray:
            """
            Generates MSDR under null hypothesis of unrelated ranks A = rank(a) and signed-ranks B, weighted by
            w = a / mean(a), with number of positive (good) ranks in B picked from binomial(p, alpha). With alpha=0.5,
            positive (good) ranks are as likely as negative (bad) ranks.
            """
            _B = np.arange(1, p + 1)
            msdr_w = np.empty(n_samples)
            p_good = np.random.binomial(p, alpha, n_samples)
            shuffle_idx_a = np.argsort(np.random.uniform(-1, 1, (n_samples, p)), axis=1)
            shuffle_idx_b = np.argsort(np.random.uniform(-1, 1, (n_samples, p)), axis=1)
            for i in range(n_samples):
                n_good = p_good[i]
                n_bad = p - n_good
                B = np.hstack((_B[:n_good], -_B[:n_bad]))[shuffle_idx_b[i]]
                msdr_w[i] = np.mean(w[shuffle_idx_a[i]] * (A[shuffle_idx_a[i]] - B) ** 2)
            return msdr_w

    def __init__(self, is_vi: pd.Series, oos_pbsv: pd.Series, pbsv_loss_type: LossType):
        """
        :param is_vi: In-sample variable importance measure
        :param oos_pbsv: Out-of-sample performance-based Shapley values
        :param pbsv_loss_type: Type of loss used for PBSV (higher is better or lower is better)
        """

        assert is_vi.shape == oos_pbsv.shape
        assert ~is_vi.index.has_duplicates
        assert is_vi.index.isin(oos_pbsv.index).all()

        oos_pbsv = oos_pbsv.reindex(is_vi.index)

        if "base_contribution" in is_vi:
            is_vi = is_vi.drop("base_contribution")
            oos_pbsv = oos_pbsv.drop("base_contribution")
            print("WARN: 'base_contribution' was removed from VI and PBSV")

        if (is_vi == 0).any():
            is_vi, oos_pbsv = is_vi.loc[is_vi > 0], oos_pbsv[is_vi > 0]
            print(
                "WARN: VI contains zero contributions, removing unused predictors results in new p=%i" % is_vi.shape[0]
            )

        assert is_vi.shape[0] > 0

        self._p = is_vi.shape[0]
        self._vi = is_vi
        self._pbsv = oos_pbsv
        self._loss_type = pbsv_loss_type

    def compute(self, mas_type: MASType = MASType.IMPORTANCE_WEIGHTED, hypothesis_test: bool = True,
                h0_alpha: float = 0.5, n_samples: int = 1000000) -> dict:
        """
        :param mas_type: Importance-weighted (recommended) assigns weights proportionally to in-sample weights so that
        ranks of important features matter more in the MAS.
        :param hypothesis_test: Compute the p-value of observing a MAS at least as extreme as observed under the null
        of unrelated ranks at 'h0_alpha'.
        :param h0_alpha: The proportion of good predictors under the null. At 'h0_alpha = 0.5', good predictors are as
        likely as bad predictors. Only relevant when 'hypothesis_test = True'.
        :param n_samples: The number of samples to draw from the null. Only relevant when 'hypothesis_test = True'.
        """
        assert 0 <= h0_alpha <= 1
        assert n_samples > 0

        vi_ranked = MAS._rank(self._vi)
        pbsv_signed_ranked = MAS._signed_rank(self._pbsv, self._loss_type)

        msdr, mas, p_value = None, None, None

        if mas_type == MAS.MASType.EQUAL_WEIGHTED:
            msdr = float(np.mean((vi_ranked - pbsv_signed_ranked) ** 2))

            if hypothesis_test:
                h0 = MAS.H0(p=self._p, mas_type=mas_type, n_samples=n_samples, alpha=h0_alpha)
                p_value = h0.get_p_value(msdr)

            expected_msdr = MAS._expected_msdr(self._p)
            mas = 1 - msdr / expected_msdr

        elif mas_type == MAS.MASType.IMPORTANCE_WEIGHTED:
            w = self._vi / self._vi.mean()
            msdr = float(np.mean(w * (vi_ranked - pbsv_signed_ranked) ** 2))

            if hypothesis_test:
                h0_w = MAS.H0(
                    p=self._p, mas_type=mas_type, n_samples=n_samples, alpha=h0_alpha,
                    is_vi_w=w.to_numpy(), is_vi_r=vi_ranked.to_numpy()
                )
                p_value = h0_w.get_p_value(msdr)

            expected_msdr = MAS._expected_msdr_w(w, vi_ranked)
            mas = 1 - msdr / expected_msdr

        if hypothesis_test:
            return {
                "mas": mas,
                "mas_p_value": p_value
            }
        else:
            return {
                "mas": mas,
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
