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
"""Settings for the surrogate-based optimization algorithm."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TCH003
from pathlib import Path  # noqa: TCH003

from gemseo.algos.base_driver_library import DriverLibrarySettingType  # noqa: TCH002
from gemseo.algos.opt.base_optimization_library_settings import (  # noqa: TCH002
    BaseOptimizationLibrarySettings,
)
from gemseo.mlearning.core.algos.ml_algo import MLAlgoParameterType  # noqa: TCH002
from gemseo.mlearning.regression.algos.base_regressor import (  # noqa: TCH002
    BaseRegressor,
)
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from pydantic import Field
from pydantic import NonNegativeInt


class SBOSettings(BaseOptimizationLibrarySettings):
    """The settings for the surrogate-based optimization algorithm."""

    acquisition_algorithm: str = Field(
        default="",
        description=(
            """The name of the algorithm to optimize the data acquisition criterion.
            If empty, use the default algorithm with its default settings."""
        ),
    )

    acquisition_settings: Mapping[str, DriverLibrarySettingType] = Field(
        default_factory=dict,
        description=(
            """The settings of the algorithm
            to optimize the data acquisition criterion.
            Ignored when `acquisition_algorithm` is empty."""
        ),
    )

    doe_algorithm: str = Field(
        default="OT_OPT_LHS",
        description=(
            """The name of the DOE algorithm for the initial sampling.

            This argument is ignored
            when regression_algorithm is a
            [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            """
        ),
    )

    doe_settings: Mapping[str, DriverLibrarySettingType] = Field(
        default_factory=dict,
        description=(
            """The settings of the DOE algorithm for the initial sampling.

            This argument is ignored
            when regression_algorithm is a
            [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            """
        ),
    )

    doe_size: NonNegativeInt = Field(
        default=10,
        description=(
            """Either the initial DOE size or 0 if it is inferred from `doe_settings`.

            This argument is ignored
            when regression_algorithm is a
            [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            """
        ),
    )

    normalize_design_space: bool = Field(
        default=False,
        description=(
            """Whether to normalize the design space variables between 0 and 1."""
        ),
    )

    regression_algorithm: str | BaseRegressor = Field(
        default=OTGaussianProcessRegressor.__name__,
        description=(
            """The regression algorithm.

            Either the name of the regression algorithm
            approximating the objective function over the design space
            or the regression algorithm itself.
            """
        ),
    )

    regression_file_path: str | Path = Field(
        default="",
        description=(
            """The path to the file to save the regression model.

            If empty, do not save the regression model.
            """
        ),
    )

    regression_settings: Mapping[str, MLAlgoParameterType] = Field(
        default_factory=dict,
        description=(
            """The settings of the regression algorithm.

            This argument is ignored
            when regression_algorithm is a
            [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            """
        ),
    )
