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

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.driver_library import DriverLibraryOptionType
from gemseo.algos.opt.optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.optimization_library import OptimizationLibrary
from gemseo.mlearning.core.algos.ml_algo import MLAlgoParameterType
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_mlearning.algos.opt import OptimizationLibraryOptionType
from gemseo_mlearning.algos.opt.core.surrogate_based_optimizer import (
    SurrogateBasedOptimizer,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.algos.opt_result import OptimizationResult


SBOOptionType = Union[
    int,
    float,
    str,
    Mapping[str, DriverLibraryOptionType],
    Mapping[str, MLAlgoParameterType],
    Mapping[str, Any],
    Mapping[str, OptimizationLibraryOptionType],
]


@dataclass
class SurrogateBasedAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a surrogate-based optimization algorithm."""

    library_name: str = "gemseo-mlearning"


class SurrogateBasedOptimization(OptimizationLibrary):
    """A wrapper for surrogate-based optimization."""

    LIBRARY_NAME = SurrogateBasedAlgorithmDescription.library_name
    __SBO = "SBO"

    _NORMALIZE_DS: ClassVar[bool] = False

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.descriptions = {
            self.__SBO: SurrogateBasedAlgorithmDescription(
                algorithm_name=self.__SBO,
                description="GEMSEO in-house surrogate-based optimizer.",
                handle_equality_constraints=False,
                handle_inequality_constraints=False,
                handle_integer_variables=True,  # provided acquisition handles integers
                internal_algorithm_name=self.__SBO,
            )
        }

    def _get_options(
        self,
        max_iter: int = 99,
        max_time: float = 0.0,
        ftol_rel: float = 1e-8,
        ftol_abs: float = 1e-14,
        xtol_rel: float = 1e-8,
        xtol_abs: float = 1e-14,
        stop_crit_n_x: int = 3,
        doe_size: int = 10,
        doe_algorithm: str = OpenTURNS.OT_LHSO,
        doe_options: Mapping[str, DriverLibraryOptionType] = READ_ONLY_EMPTY_DICT,
        regression_algorithm: (str | BaseRegressor) = GaussianProcessRegressor.__name__,
        regression_options: Mapping[str, MLAlgoParameterType] = READ_ONLY_EMPTY_DICT,
        regression_file_path: str | Path = "",
        acquisition_algorithm: str = "DIFFERENTIAL_EVOLUTION",
        acquisition_options: Mapping[
            str, OptimizationLibraryOptionType
        ] = READ_ONLY_EMPTY_DICT,
        **kwargs: Any,
    ) -> dict:
        """Set the default options values.

        Args:
            max_iter: The maximum number of evaluations.
            max_time: The maximum runtime in seconds.
                The value 0 disables the cap on the runtime.
            ftol_rel: The relative tolerance on the objective function.
            ftol_abs: The absolute tolerance on the objective function.
            xtol_rel: The relative tolerance on the design parameters.
            xtol_abs: The absolute tolerance on the design parameters.
            stop_crit_n_x: The number of design vectors to take into account in the
                stopping criteria.
            normalize_design_space: Whether to normalize the design variables between 0
                and 1.
            doe_size: Either the size of the initial DOE
                or `0` if the size is inferred from `doe_options`.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            doe_algorithm: The name of the algorithm for the initial sampling.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            doe_options: The options of the algorithm for the initial sampling.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            regression_algorithm: Either the name of the regression algorithm
                approximating the objective function over the design space
                or the regression algorithm itself.
            regression_file_path: The path to the file to save the regression model.
                If empty, do not save the regression model.
            regression_options: The options of the regression algorithm.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            acquisition_algorithm: The name of the algorithm to optimize the data
                acquisition criterion.
            acquisition_options: The options of the algorithm to optimize
                the data acquisition criterion.
            **kwargs: Other driver options.

        Returns:
            The processed options.
        """
        return self._process_options(
            max_iter=max_iter,
            max_time=max_time,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            stop_crit_n_x=stop_crit_n_x,
            doe_size=doe_size,
            doe_algorithm=doe_algorithm,
            doe_options=doe_options,
            regression_algorithm=regression_algorithm,
            regression_options=regression_options,
            regression_file_path=regression_file_path,
            acquisition_algorithm=acquisition_algorithm,
            acquisition_options=acquisition_options,
            **kwargs,
        )

    def _run(self, **options: SBOOptionType) -> OptimizationResult:
        """
        Raises:
            ValueError: When the maximum number of iterations
                is less than or equal to the initial DOE size.
        """  # noqa: D205 D212 D415
        doe_options = options["doe_options"]
        doe_size = options["doe_size"]
        doe_algorithm = options["doe_algorithm"]
        regression_algorithm = options["regression_algorithm"]
        if not isinstance(regression_algorithm, BaseRegressor):
            # The number of evaluations is equal to
            #     1 for the initial evaluation in OptimizationLibrary._pre_run
            #   + N for the N-length DOE
            #   + n_iter - 1 - N
            # So, n_iter - 1 - N >= 0 implies that n_iter >= 1+N
            doe_algo = DOELibraryFactory().create(doe_algorithm)
            initial_doe_size = len(
                doe_algo.compute_doe(self.problem.design_space, doe_size, **doe_options)
            )
            max_iter = options[self.MAX_ITER]
            if max_iter < 1 + initial_doe_size:
                msg = (
                    f"max_iter ({max_iter}) must be "
                    f"strictly greater than the initial DOE size ({initial_doe_size})."
                )
                raise ValueError(msg)

        optimizer = SurrogateBasedOptimizer(
            self.problem,
            options["acquisition_algorithm"],
            doe_size=doe_size,
            doe_algorithm=doe_algorithm,
            doe_options=doe_options,
            regression_algorithm=regression_algorithm,
            regression_options=options["regression_options"],
            regression_file_path=options["regression_file_path"],
            acquisition_options=options["acquisition_options"],
        )
        # Set a large bound on the number of acquisitions as GEMSEO handles stopping
        return self.get_optimum_from_database(optimizer.execute(sys.maxsize))
