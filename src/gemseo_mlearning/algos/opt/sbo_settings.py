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

from enum import StrEnum
from pathlib import Path  # noqa: TC003

from gemseo.core.algorithm.base_driver_settings import BaseDriverSettings  # noqa: TC002
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.doe.core.base_doe_settings import BaseDOESettings  # noqa: TC002
from gemseo.machine_learning.regression.core.base_regressor import (  # noqa: TC002
    BaseRegressor,
)
from gemseo.machine_learning.regression.core.base_regressor_settings import (  # noqa: TC002
    BaseRegressorSettings,
)
from gemseo.machine_learning.regression.model.ot_gpr_settings import (
    OTGaussianProcessRegressor_Settings,
)
from gemseo.optimization.core.base_optimizer_settings import BaseOptimizerSettings
from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002


def create_default_doe_settings() -> OT_OPT_LHS_Settings:
    """Create the default settings of the DOE algorithm for the initial sampling.

    Returns:
        The settings of an optimized LHS with 10 samples.
    """
    return OT_OPT_LHS_Settings(n_samples=10)


class AcquisitionCriterion(StrEnum):
    r"""An acquisition criterion.

    In the following,
    the training output values already used
    and the random output of the surrogate model at a given input point $x$
    are respectively denoted $\{y_1,\ldots,y_n\}$ and $Y(x)$.
    The expectation and the standard deviation of $Y(x)$ are respectively denoted
    $\mathbb{E}[Y(x)]$ and $\mathbb{S}[Y(x)]$.
    """

    EI = "EI"
    r"""The expected improvement.

    The acquisition criterion is $\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y(x),0]$.
    """

    CB = "CB"
    r"""The confidence bound.

    The acquisition criterion is $\mathbb{E}[Y(x)]-3\mathbb{S}[Y(x)]$.
    """

    Output = "Output"
    r"""The mean output.

    The acquisition criterion is $\mathbb{E}[Y(x)]$.
    """


class SBO_Settings(BaseOptimizerSettings):  # noqa: N801
    """The settings for the surrogate-based optimization algorithm."""

    acquisition_settings: BaseDriverSettings | None = Field(
        default=None,
        description=(
            """The settings of the algorithm
            to optimize the data acquisition criterion.
            If `None`, use the default algorithm with its default settings."""
        ),
    )

    batch_size: PositiveInt = Field(
        default=1, description="The number of points to be acquired in parallel."
    )

    criterion: AcquisitionCriterion = Field(
        default=AcquisitionCriterion.EI, description="The acquisition criterion."
    )

    doe_settings: BaseDOESettings = Field(
        default_factory=create_default_doe_settings,
        description=(
            """The settings of the DOE algorithm for the initial sampling.
            This argument is ignored
            when regressor is a
            [BaseRegressor][gemseo.machine_learning.regression.core.base_regressor.BaseRegressor].
            """
        ),
    )

    mc_size: PositiveInt = Field(
        default=10_000,
        description="The sample size to estimate the acquisition criteria in parallel.",
    )

    normalize_design_space: bool = Field(
        default=False,
        description=(
            """Whether to normalize the design space variables between 0 and 1."""
        ),
    )

    regressor: BaseRegressorSettings | BaseRegressor = Field(
        default_factory=OTGaussianProcessRegressor_Settings,
        description=(
            """The regressor to approximate the objective function.
            Either a regressor or regressor settings.
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
