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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Criterion to be minimized for estimating a quantile, with exploration.

Statistic:

$$EI[x] = \mathbb{E}[|q(\alpha)-Y(x)|]$$

where $q$ is a quantile with level $\alpha$.

Bootstrap estimator:

$$\widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B |q-Y_b(x)|$$
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.utils.string_tools import pretty_str
from numpy import quantile

from gemseo_mlearning.active_learning.acquisition_criteria.level_set import LevelSet

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class Quantile(LevelSet):
    """Criterion to be minimized for estimating a quantile, with exploration."""

    def __init__(
        self,
        algo_distribution: BaseRegressorDistribution,
        level: float,
        uncertain_space: ParameterSpace,
        n_samples: int = 10000,
    ) -> None:
        """
        Args:
            level: The quantile level.
            uncertain_space: The uncertain variable space.
            n_samples: The number of samples
                to estimate the quantile of the regression model by Monte Carlo.
        """  # noqa: D205 D212 D415
        input_names = algo_distribution.input_names
        missing_names = set(input_names) - set(uncertain_space.variable_names)
        if missing_names:
            msg = (
                "The probability distributions of the input variables "
                f"{pretty_str(missing_names, use_and=True)} are missing."
            )
            raise ValueError(msg)

        # Create a new uncertain space sorted by model inputs.
        new_uncertain_space = uncertain_space.__class__()
        for name in input_names:
            new_uncertain_space[name] = uncertain_space[name]

        data = algo_distribution.predict(new_uncertain_space.compute_samples(n_samples))
        super().__init__(algo_distribution, quantile(data, level))
