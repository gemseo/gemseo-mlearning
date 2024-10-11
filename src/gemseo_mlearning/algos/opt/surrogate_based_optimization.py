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
"""A library for surrogate-based optimization."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union

from gemseo.algos.base_driver_library import DriverLibrarySettingType
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.mlearning.core.algos.ml_algo import MLAlgoParameterType
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor

from gemseo_mlearning.algos.opt.core.surrogate_based_optimizer import (
    SurrogateBasedOptimizer,
)
from gemseo_mlearning.algos.opt.sbo_settings import SBOSettings

if TYPE_CHECKING:
    from gemseo.algos.base_problem import BaseProblem
    from gemseo.algos.optimization_result import OptimizationResult


SBOSettingType = Union[
    int,
    float,
    str,
    Mapping[str, DriverLibrarySettingType],
    Mapping[str, MLAlgoParameterType],
    Mapping[str, Any],
]


@dataclass
class SurrogateBasedAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a surrogate-based optimization algorithm."""

    library_name: str = "gemseo-mlearning"


class SurrogateBasedOptimization(BaseOptimizationLibrary):
    """A wrapper for surrogate-based optimization."""

    LIBRARY_NAME = SurrogateBasedAlgorithmDescription.library_name

    _NORMALIZE_DS: ClassVar[bool] = False
    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "SBO": SurrogateBasedAlgorithmDescription(
            algorithm_name="SBO",
            description="GEMSEO in-house surrogate-based optimizer.",
            handle_equality_constraints=False,
            handle_inequality_constraints=False,
            handle_integer_variables=True,  # provided acquisition handles integers
            internal_algorithm_name="SBO",
            settings=SBOSettings,
        )
    }

    def _run(
        self, problem: BaseProblem, **settings: SBOSettingType
    ) -> OptimizationResult:
        """
        Raises:
            ValueError: When the maximum number of iterations
                is less than or equal to the initial DOE size.
        """  # noqa: D205 D212 D415
        doe_settings = settings["doe_settings"]
        doe_size = settings["doe_size"]
        doe_algorithm = settings["doe_algorithm"]
        regression_algorithm = settings["regression_algorithm"]
        if not isinstance(regression_algorithm, BaseRegressor):
            # The number of evaluations is equal to
            #     1 for the initial evaluation in OptimizationLibrary._pre_run
            #   + N for the N-length DOE
            #   + n_iter - 1 - N
            # So, n_iter - 1 - N >= 0 implies that n_iter >= 1+N
            doe_algo = DOELibraryFactory().create(doe_algorithm)
            initial_doe_size = len(
                doe_algo.compute_doe(
                    self.problem.design_space, doe_size, **doe_settings
                )
            )
            max_iter = settings["max_iter"]
            if max_iter < 1 + initial_doe_size:
                msg = (
                    f"max_iter ({max_iter}) must be "
                    f"strictly greater than the initial DOE size ({initial_doe_size})."
                )
                raise ValueError(msg)

        optimizer = SurrogateBasedOptimizer(
            problem,
            settings["acquisition_algorithm"],
            doe_size=doe_size,
            doe_algorithm=doe_algorithm,
            doe_settings=doe_settings,
            regression_algorithm=regression_algorithm,
            regression_options=settings["regression_options"],
            regression_file_path=settings["regression_file_path"],
            **settings["acquisition_settings"],
        )
        # Set a large bound on the number of acquisitions as GEMSEO handles stopping
        return self._get_optimum_from_database(problem, optimizer.execute(sys.maxsize))
