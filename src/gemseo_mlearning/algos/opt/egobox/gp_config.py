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
"""Gaussian process model configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt

from gemseo_mlearning.algos.opt.egobox.correlation_spec import CorrelationSpec
from gemseo_mlearning.algos.opt.egobox.regression_spec import RegressionSpec


class GpConfig(BaseModel):
    """The Gaussian process (GP) model configuration."""

    CorrelationSpec: ClassVar[type[CorrelationSpec]] = CorrelationSpec
    RegressionSpec: ClassVar[type[RegressionSpec]] = RegressionSpec

    regr_spec: RegressionSpec = Field(
        default=RegressionSpec.CONSTANT,
        description="The type of trend for the GP models.",
    )

    corr_spec: CorrelationSpec = Field(
        default=CorrelationSpec.SQUARED_EXPONENTIAL,
        description="The type of correlation kernel for the GP models.",
    )

    kpls_dim: NonNegativeInt | None = Field(
        default=None,
        description="The number of components "
        "for the KPLS dimension reduction technique "
        "(Kriging stands for Kriging and PLS for Partial Least Squares). "
        "This number must be less than or equal to the input dimension. "
        "This is used to address high-dimensional problems, "
        "typically when the input dimension is greater than 10."
        "If `None`, do not use the KPLS dimension reduction technique.",
    )

    n_clusters: int = Field(
        default=1,
        description="The number of clusters used by the mixture of surrogate experts."
        "When set to 0, "
        "the number of clusters is determined automatically "
        "and refreshed every 10-points addition "
        "(should say 'tentative addition' "
        "because addition may fail for some points but it is counted anyway)."
        "When set to negative number `-n`, "
        "the number of clusters is determined automatically in `[1, n]` "
        "this is used to limit the number of trials, hence the execution time.",
    )

    use_smooth_moe: bool = Field(
        default=False,
        description="Whether to use a smooth mixture of exports (MOE)."
        "If `True`, "
        "the MOE combines experts prediction wrt their responsibilities,"
        "the Heaviside factor which controls steepness of the change "
        "between experts regions is optimized to get the best mixture quality."
        "If `False, "
        "the MOE uses the prediction of the expert "
        "with the highest responsibility resulting in a model with discontinuities.",
    )

    theta_init: dict[str, list[float]] = Field(
        default_factory=dict,
        description="The initial guess for the GP theta hyperparameters "
        "related to the different design variables."
        "The default is `default_theta_init` for the design variables not specified.",
    )

    theta_bounds: dict[str, tuple[list[float], list[float]]] = Field(
        default_factory=dict,
        description="The space search when optimizing theta GP hyperparameters,"
        "of the form `{name: [lower, upper], ...}`"
        "where `nx` is the input dimension."
        "The default is `default_theta_bounds` for the design variables not specified.",
    )

    n_start: int = Field(
        default=10,
        description="The number of internal GP hyperparameters optimization restart."
        "When is negative, optimization is disabled and theta initial value is used.",
    )

    max_eval: NonNegativeInt = Field(
        default=50,
        description="The maximum number of likelihood evaluations "
        "during GP hyperparameters optimization.",
    )

    default_theta_init: float = Field(
        default=1e-1,
        description="The default initial guess for the GP theta hyperparameters.",
    )

    default_theta_bounds: tuple[float, float] = Field(
        default=(1e-2, 1e1),
        description="The default bounds for the GP theta hyperparameters.",
    )
