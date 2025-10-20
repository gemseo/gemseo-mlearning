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
"""Settings for the SMT surrogate-based optimization algorithms."""

from __future__ import annotations

from enum import auto

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from smt.surrogate_models.surrogate_model import SurrogateModel  # noqa: TC002
from strenum import StrEnum


class AcquisitionCriterion(StrEnum):
    r"""An acquisition criterion.

    In the following, the training output values already used and the output and
    uncertainty predictions at a given input point $x$ are respectively denoted
    $\{y_1,\ldots,y_n\}$, $\mu(x)$ and $\sigma(x)$.
    """

    EI = auto()
    r"""The expected improvement.

    The acquisition criterion is $\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y,0]$ where $Y$ is
    a Gaussian random variable with mean $\mu(x)$ and standard deviation $\sigma(x)$.
    """

    LCB = auto()
    r"""The lower confidence bound.

    The acquisition criterion is $\mu(x)-3\sigma(x)$.
    """

    SBO = auto()
    r"""The surrogate-based optimization.

    The acquisition criterion is $\mu(x)$.
    """


class ParallelStrategy(StrEnum):
    r"""The strategy to set the outputs of the virtual points for parallel acquisition.

    In the following, the output variable, the training output values already used, and
    the output and uncertainty predictions at a given input point $x$ are respectively
    denoted $y$, $\{y_1,\ldots,y_n\}$, $\mu(x)$ and $\sigma(x)$.
    """

    CLmin = auto()
    r"""The minimum constant liar.

    The output of the virtual point at $x$ is defined by $\min \{y_1,\ldots,y_n\}$.
    """

    KB = auto()
    r"""The Kriging believer.

    The output of the virtual point at $x$ is defined by $\mu(x)$.
    """

    KBLB = auto()
    r"""The Kriging believer lower bound.

    The output of the virtual point at $x$ is defined by $\mu(x)-3\sigma(x)$.
    """

    KBRand = auto()
    r"""The Kriging believer random bound.

    The output of the virtual point at $x$ is defined by $\mu(x)+\kappa(x)\sigma(x)$
    where $\kappa(x)$ is the realization of a random variable distributed according to
    the standard normal distribution.
    """

    KBUB = auto()
    r"""The Kriging believer upper bound.

    The output of the virtual point at $x$ is defined by $\mu(x)+3\sigma(x)$.
    """


class Surrogate(StrEnum):
    """A surrogate model."""

    GPX = auto()
    """Kriging based on the `egobox` library."""

    KRG = auto()
    """Kriging."""

    KPLS = auto()
    """Kriging using partial least squares (PLS) to reduce the input dimension."""

    KPLSK = auto()
    """A variant of KPLS."""

    MGP = auto()
    """A marginal Gaussian process (MGP) regressor."""


class SMT_EGO_Settings(BaseOptimizerSettings):  # noqa: N801
    """The settings of the SMT's SBO algorithm."""

    criterion: AcquisitionCriterion = Field(
        default=AcquisitionCriterion.EI, description="The acquisition criterion."
    )

    enable_tunneling: bool = Field(
        default=False,
        description=(
            "Whether to enable the penalization of points "
            "that have been already evaluated in EI criterion."
        ),
    )

    n_doe: PositiveInt = Field(
        default=10,
        description=(
            "The number of points of the initial LHS DOE to train the surrogate."
        ),
    )

    n_max_optim: PositiveInt = Field(
        default=20,
        description="The maximum number of iterations for each sub-optimization.",
    )

    n_parallel: PositiveInt = Field(
        default=1,
        description=(
            "The number of points to be acquired in parallel; "
            "if `1`, acquire points sequentially."
        ),
    )

    n_start: PositiveInt = Field(
        default=50, description="The number of sub-optimizations."
    )

    qEI: ParallelStrategy = Field(  # noqa: N815
        default=ParallelStrategy.KBUB,
        description="The strategy to acquire points in parallel.",
    )

    random_state: NonNegativeInt = Field(
        default=1,
        description=(
            "The Numpy RandomState object or seed number which controls random draws."
        ),
    )

    surrogate: Surrogate | SurrogateModel = Field(
        default=Surrogate.KRG,
        description=(
            "The SMT Kriging-based surrogate model used internally; "
            "either an instance of the surrogate before training or its class name."
        ),
    )


# TODO: API: remove this alias.
SMTEGOSettings = SMT_EGO_Settings
