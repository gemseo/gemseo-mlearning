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
"""Settings for the EGObox_Egor algorithm."""

from __future__ import annotations

from math import inf
from typing import ClassVar

from gemseo.algos.doe.base_doe_settings import BaseDOESettings  # noqa: TC002
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from numpy import ndarray  # noqa: TC002
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from gemseo_mlearning.algos.opt.egobox.constraint_strategy import ConstraintStrategy
from gemseo_mlearning.algos.opt.egobox.gp_config import GpConfig
from gemseo_mlearning.algos.opt.egobox.infill_optimizer import InfillOptimizer
from gemseo_mlearning.algos.opt.egobox.infill_strategy import InfillStrategy
from gemseo_mlearning.algos.opt.egobox.q_infill_strategy import QInfillStrategy


class EGObox_Egor_Settings(BaseOptimizerSettings):  # noqa: N801
    """The settings of the EGObox_Egor algorithm."""

    _TARGET_CLASS_NAME = "EGObox_Egor"

    ConstraintStrategy: ClassVar[type[ConstraintStrategy]] = ConstraintStrategy
    GpConfig: ClassVar[type[GpConfig]] = GpConfig
    InfillOptimizer: ClassVar[type[InfillOptimizer]] = InfillOptimizer
    InfillStrategy: ClassVar[type[InfillStrategy]] = InfillStrategy
    QInfillStrategy: ClassVar[type[QInfillStrategy]] = QInfillStrategy

    gp_config: GpConfig = Field(
        default=GpConfig(),
        description="The Gaussian process configuration used by the optimizer.",
    )

    n_start: PositiveInt = Field(
        default=20,
        description="The number of runs of the infill strategy optimizations "
        "(the best result is taken).",
    )

    n_doe: NonNegativeInt = Field(
        default=10,
        description="The number of samples of the initial LHS sampling."
        "When `0`, a number of points is computed automatically "
        "regarding the number of input variables"
        "of the function under optimization."
        "This option is ignore when `doe` is provided.",
    )

    doe: ndarray | BaseDOESettings | None = Field(
        default=None,
        description="Either the initial DOE or DOE algorithm settings."
        "If `None`, a DOE is created using the `n_doe` option.",
    )

    infill_strategy: InfillStrategy = Field(
        default=InfillStrategy.WB2,
        description="The infill criterion to decide the best next promising point.",
    )

    infill_optimizer: InfillOptimizer = Field(
        default=InfillOptimizer.COBYLA,
        description="The internal optimizer used to optimize the infill criterion.",
    )

    cstr_strategy: ConstraintStrategy = Field(
        default=ConstraintStrategy.MC,
        description="The constraint strategy.",
    )

    q_infill_strategy: QInfillStrategy = Field(
        default=QInfillStrategy.KB,
        description="The parallel infill criterion (a.k.a. qEI) "
        "to get virtual next promising points "
        "in order to allow `q` parallel evaluations of the function under optimization "
        "(only used when `q_points > 1`).",
    )

    q_points: PositiveInt = Field(
        default=1,
        description="The number of points to be evaluated "
        "to allow parallel evaluation of the function under optimization.",
    )

    q_optmod: NonNegativeInt = Field(
        default=1,
        description="The number of iterations "
        "between two surrogate models true training "
        "(i.e., including hyperparameters optimization);"
        "otherwise previous hyperparameters are re-used only "
        "when computing `q_points` to be evaluated in parallel."
        "The default value is 1 meaning surrogates are properly trained "
        "for each `q` points determination."
        "The value is used "
        "as a modulo of `iteration_number * q_points` to trigger true training."
        "This is used to decrease the number of training "
        "at the expense of surrogate accuracy.",
    )

    trego: bool = Field(
        default=False,
        description="When `True`, TREGO algorithm is used, "
        "otherwise classic EGO algorithm is used.",
    )

    coego_n_coop: NonNegativeInt = Field(
        default=0,
        description="The number of cooperative components groups "
        "which will be used by the CoEGO algorithm."
        "It is better to have `n_coop` a divider of `nx` "
        "or if not with a remainder as large as possible."
        "The CoEGO algorithm is used "
        "to tackle high-dimensional problems turning it "
        "in a set of partial optimizations "
        "using only `nx / n_coop components` at a time."
        "The default value is `0` meaning that the CoEGO algorithm is not used.",
    )

    target: float = Field(
        default=-inf, description="The known optimum used as stopping criterion."
    )

    outdir: str | None = Field(
        default=None,
        description="The directory to write optimization history "
        "and used as search path for warm start `doe`.",
    )

    warm_start: bool = Field(
        default=False,
        description="Start by loading initial `doe` from `outdir` directory.",
    )

    hot_start: NonNegativeInt | None = Field(
        default=None,
        description="When `hot_start>=0` saves optimizer state at each iteration "
        "and starts from a previous checkpoint"
        "if any for the given hot_start number of iterations "
        "beyond the `max_iters` nb of iterations."
        "Ignored if `hot_start` is `None`."
        "In an unstable environment were there can be crashes "
        "it allows to restart the optimization"
        "from the last iteration till stopping criterion is reached. "
        "Just use `hot_start=0` in this case."
        "When specifying an extended nb of iterations (`hot_start > 0`) "
        "it can allow to continue till `max_iters +hot_start` number of iterations"
        " is reached (provided the stopping criterion is `max_iters`)"
        "Checkpoint information is stored in `.checkpoint/egor.arg` binary file.",
    )

    seed: NonNegativeInt | None = Field(
        default=0,
        description="The random generator seed to allow computation reproducibility.",
    )
