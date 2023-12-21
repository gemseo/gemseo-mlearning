# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""The distributions of machine learning algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_mlearning.adaptive.distributions.kriging_distribution import (
    KrigingDistribution,
)
from gemseo_mlearning.adaptive.distributions.regressor_distribution import (
    RegressorDistribution,
)

if TYPE_CHECKING:
    from gemseo.mlearning.regression.regression import MLRegressionAlgo

    from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution


def get_regressor_distribution(
    regression_algorithm: MLRegressionAlgo,
    use_bootstrap: bool = True,
    use_loo: bool = False,
    size: int | None = None,
) -> MLRegressorDistribution:
    """Return the distribution of a regression algorithm.

    Args:
        regression_algorithm: The regression algorithm.
        bootstrap: Whether to use bootstrap for resampling.
            If ``False``, use cross-validation.
        use_loo: Whether to use leave-one-out resampling when
            ``use_bootstrap`` is ``False``.
            If ``False``, use parameterized cross-validation.
        size: The size of the resampling set,
            i.e. the number of times the regression algorithm is rebuilt.
            If ``None``, use the default size
            for bootstrap (:attr:`.MLAlgoSampler.N_BOOTSTRAP`)
            and cross-validation (:attr:`.MLAlgoSampler.N_FOLDS`).
            This argument does not apply to leave-one-out.

    Returns:
        The distribution of the regression algorithm.
    """
    if hasattr(regression_algorithm, "predict_std"):
        return KrigingDistribution(regression_algorithm)

    return RegressorDistribution(regression_algorithm, use_bootstrap, use_loo, size)
