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
"""Regressor distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.mlearning.regression.algos.base_random_process_regressor import (
    BaseRandomProcessRegressor,
)

from gemseo_mlearning.active_learning.distributions.kriging_distribution import (
    KrigingDistribution,
)
from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
    RegressorDistribution,
)

if TYPE_CHECKING:
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


def get_regressor_distribution(
    regressor: BaseRegressor,
    use_bootstrap: bool = True,
    use_loo: bool = False,
    size: int | None = None,
) -> BaseRegressorDistribution:
    """Return the distribution of a regressor.

    Args:
        regressor: The regression algorithm.
        use_bootstrap: Whether to use bootstrap for resampling.
            If `False`, use cross-validation.
        use_loo: Whether to use leave-one-out resampling when
            `use_bootstrap` is `False`.
            If `False`, use parameterized cross-validation.
        size: The size of the resampling set,
            i.e. the number of times the regression algorithm is rebuilt.
            If `None`,
            [N_BOOTSTRAP][gemseo_mlearning.active_learning.distributions.regressor_distribution.RegressorDistribution.N_BOOTSTRAP]
            in the case of bootstrap
            and
            [N_FOLDS][gemseo_mlearning.active_learning.distributions.regressor_distribution.RegressorDistribution.N_FOLDS]
            in the case of cross-validation.
            This argument is ignored in the case of leave-one-out.

    Returns:
        The distribution of the regression algorithm.
    """
    if isinstance(regressor, BaseRandomProcessRegressor):
        return KrigingDistribution(regressor)

    return RegressorDistribution(regressor, use_bootstrap, use_loo, size)
