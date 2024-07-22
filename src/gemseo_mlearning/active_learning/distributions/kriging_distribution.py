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
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Kriging-like regressor distribution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from gemseo.mlearning.data_formatters.regression_data_formatters import (
    RegressionDataFormatters,
)
from scipy.stats import norm

from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (
    BaseRegressorDistribution,
)

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.mlearning.regression.algos.base_random_process_regressor import (
        BaseRandomProcessRegressor,
    )
    from gemseo.typing import NumberArray


class KrigingDistribution(BaseRegressorDistribution):
    """Kriging-like regressor distribution."""

    def __init__(  # noqa: D107
        self, algo: BaseRandomProcessRegressor
    ) -> None:
        super().__init__(algo)

    def compute_confidence_interval(  # noqa: D102
        self,
        input_data: DataType,
        level: float = 0.95,
    ) -> (
        tuple[dict[str, NumberArray], dict[str, NumberArray]]
        | tuple[NumberArray, NumberArray]
    ):
        mean = self.compute_mean(input_data)
        std = self.compute_standard_deviation(input_data)
        quantile = norm.ppf(level)
        if isinstance(mean, Mapping):
            lower = {name: mean[name] - quantile * std[name] for name in mean}
            upper = {name: mean[name] + quantile * std[name] for name in mean}
        else:
            lower = mean - quantile * std
            upper = mean + quantile * std
        return lower, upper

    @RegressionDataFormatters.format_dict
    @RegressionDataFormatters.format_samples
    def compute_mean(  # noqa: D102
        self,
        input_data: DataType,
    ) -> DataType:
        return self.algo.predict(input_data)

    @RegressionDataFormatters.format_dict
    @RegressionDataFormatters.format_samples
    def compute_variance(  # noqa: D102
        self,
        input_data: DataType,
    ) -> DataType:
        return self.compute_standard_deviation(input_data) ** 2

    @RegressionDataFormatters.format_dict
    @RegressionDataFormatters.format_samples
    def compute_standard_deviation(  # noqa: D102
        self,
        input_data: DataType,
    ) -> DataType:
        return self.algo.predict_std(input_data)
