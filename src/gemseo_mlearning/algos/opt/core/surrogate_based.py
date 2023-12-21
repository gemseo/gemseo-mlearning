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
"""A class for surrogate-based optimization."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_library import DOELibrary
from gemseo.algos.doe.doe_library import DOELibraryOptionType
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.gpr import GaussianProcessRegressor
from gemseo.mlearning.regression.regression import MLRegressionAlgo

from gemseo_mlearning.adaptive.acquisition import MLDataAcquisition
from gemseo_mlearning.adaptive.criteria.optimum.criterion import ExpectedImprovement
from gemseo_mlearning.adaptive.distributions import get_regressor_distribution

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.mlearning.core.ml_algo import MLAlgoParameterType

    from gemseo_mlearning.algos.opt import OptimizationLibraryOptionType

LOGGER = logging.getLogger(__name__)


class SurrogateBasedOptimizer:
    """An optimizer based on surrogate models."""

    __STOP_BECAUSE_ALREADY_KNOWN = "The acquired input data is already known."
    __STOP_BECAUSE_MAX_ACQUISITIONS = "All the data acquisitions have been made."

    def __init__(
        self,
        problem: OptimizationProblem,
        acquisition_algorithm: str,
        doe_size: int = 0,
        doe_algorithm: str = OpenTURNS.OT_LHSO,
        doe_options: Mapping[str, DOELibraryOptionType] = MappingProxyType({}),
        regression_algorithm: str = GaussianProcessRegressor.__name__,
        regression_options: Mapping[str, MLAlgoParameterType] = MappingProxyType({}),
        acquisition_options: Mapping[
            str, OptimizationLibraryOptionType
        ] = MappingProxyType({}),
    ) -> None:
        """
        Args:
            acquisition_algorithm: The name of the algorithm to optimize the data
                acquisition criterion.
                N.B. this algorithm must handle integers if some of the optimization
                variables are integers.
            problem: The optimization problem.
            doe_size: The size of the initial DOE.
                Should be ``0`` if the DOE algorithm does not have a
                ``n_samples`` option.
            doe_algorithm: The name of the algorithm for the initial sampling.
            doe_options: The options of the algorithm for the initial sampling.
            regression_algorithm: The name of the regression algorithm for the
                objective function.
            regression_options: The options of the regression algorithm for the
                objective function.
            acquisition_options: The options of the algorithm to optimize
                the data acquisition criterion.
        """  # noqa: D205, D212, D415
        self.__acquisition = None
        self.__distribution = None
        self.__problem = problem
        # Initialize the surrogate model of the objective function
        # Store max_iter as it will be overwritten by DOELibrary
        max_iter = self.__problem.max_iter
        options = dict(doe_options)
        if doe_size > 0 and DOELibrary.N_SAMPLES not in options:
            options[DOELibrary.N_SAMPLES] = doe_size

        DOEFactory().execute(self.__problem, doe_algorithm, **options)
        self.__problem.max_iter = max_iter
        self.__model = RegressionModelFactory().create(
            regression_algorithm,
            data=self.__problem.to_dataset(opt_naming=False),
            transformer=MLRegressionAlgo.DEFAULT_TRANSFORMER,
            **regression_options,
        )
        self.__distribution = get_regressor_distribution(self.__model)
        self.__acquisition = MLDataAcquisition(
            ExpectedImprovement.__name__,
            self.__problem.design_space,
            self.__distribution,
        )
        self.__acquisition.set_acquisition_algorithm(
            acquisition_algorithm, **acquisition_options
        )

    def execute(self, number_of_acquisitions: int) -> str:
        """Execute the surrogate-based optimization.

        Args:
            number_of_acquisitions: The number of learning points to be acquired.

        Returns:
            The termination message.
        """
        self.__distribution.learn()
        message = self.__STOP_BECAUSE_MAX_ACQUISITIONS
        for _ in range(number_of_acquisitions):
            input_data = self.__acquisition.compute_next_input_data()
            if input_data in self.__problem.database:
                message = self.__STOP_BECAUSE_ALREADY_KNOWN
                break

            self.__problem.evaluate_functions(input_data, normalize=False)
            self.__distribution.change_learning_set(
                self.__problem.to_dataset(opt_naming=False)
            )
            self.__acquisition.update_problem()

        return message
