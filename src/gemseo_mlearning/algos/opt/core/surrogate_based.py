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
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_library import DOELibrary
from gemseo.algos.doe.doe_library import DOELibraryOptionType
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.gpr import GaussianProcessRegressor
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from numpy import hstack
from numpy import newaxis
from pandas import concat

from gemseo_mlearning.adaptive.acquisition import MLDataAcquisition
from gemseo_mlearning.adaptive.criteria.optimum.criterion import ExpectedImprovement
from gemseo_mlearning.adaptive.distributions import get_regressor_distribution

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.mlearning.core.ml_algo import MLAlgoParameterType

    from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution
    from gemseo_mlearning.algos.opt import OptimizationLibraryOptionType

LOGGER = logging.getLogger(__name__)


class SurrogateBasedOptimizer:
    """An optimizer based on surrogate models."""

    __STOP_BECAUSE_ALREADY_KNOWN = "The acquired input data is already known."
    __STOP_BECAUSE_MAX_ACQUISITIONS = "All the data acquisitions have been made."

    __acquisition: MLDataAcquisition
    """The active learning algorithm to acquire new samples to learn."""

    __dataset: IODataset
    """The original learning dataset enriched by new samples."""

    __distribution: MLRegressorDistribution
    """The distribution associated with the regression algorithm."""

    __regression_file_path: str | Path
    """The path to the file to save the regression model.

    If empty, do not save the regression model.
    """

    __problem: OptimizationProblem
    """The optimization problem."""

    def __init__(
        self,
        problem: OptimizationProblem,
        acquisition_algorithm: str,
        doe_size: int = 0,
        doe_algorithm: str = OpenTURNS.OT_LHSO,
        doe_options: Mapping[str, DOELibraryOptionType] = READ_ONLY_EMPTY_DICT,
        regression_algorithm: (
            str | MLRegressionAlgo
        ) = GaussianProcessRegressor.__name__,
        regression_options: Mapping[str, MLAlgoParameterType] = READ_ONLY_EMPTY_DICT,
        regression_file_path: str | Path = "",
        acquisition_options: Mapping[
            str, OptimizationLibraryOptionType
        ] = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            acquisition_algorithm: The name of the algorithm to optimize the data
                acquisition criterion.
                N.B. this algorithm must handle integers if some of the optimization
                variables are integers.
            problem: The optimization problem.
            doe_size: Either the size of the initial DOE
                or ``0`` if the size is inferred from ``doe_options``.
                This argument is ignored
                when ``regression_algorithm`` is an :class:`.MLRegressionAlgo`.
            doe_algorithm: The name of the algorithm for the initial sampling.
                This argument is ignored
                when ``regression_algorithm`` is an :class:`.MLRegressionAlgo`.
            doe_options: The options of the algorithm for the initial sampling.
                This argument is ignored
                when ``regression_algorithm`` is an :class:`.MLRegressionAlgo`.
            regression_algorithm: Either the name of the regression algorithm
                approximating the objective function over the design space
                or the regression algorithm itself.
            regression_options: The options of the regression algorithm.
                This argument is ignored
                when ``regression_algorithm`` is an :class:`.MLRegressionAlgo`.
            regression_file_path: The path to the file to save the regression model.
                If empty, do not save the regression model.
            acquisition_options: The options of the algorithm to optimize
                the data acquisition criterion.
        """  # noqa: D205, D212, D415
        self.__problem = problem
        if isinstance(regression_algorithm, MLRegressionAlgo):
            self.__dataset = regression_algorithm.learning_set
        else:
            # Store max_iter as it will be overwritten by DOELibrary
            max_iter = self.__problem.max_iter
            options = dict(doe_options)
            if doe_size > 0 and DOELibrary.N_SAMPLES not in options:
                options[DOELibrary.N_SAMPLES] = doe_size

            DOEFactory().execute(self.__problem, doe_algorithm, **options)
            self.__dataset = self.__problem.to_dataset(opt_naming=False)
            regression_algorithm = RegressionModelFactory().create(
                regression_algorithm,
                data=self.__dataset,
                transformer=MLRegressionAlgo.DEFAULT_TRANSFORMER,
                **regression_options,
            )
            self.__problem.max_iter = max_iter

        self.__distribution = get_regressor_distribution(regression_algorithm)
        self.__acquisition = MLDataAcquisition(
            ExpectedImprovement.__name__,
            self.__problem.design_space,
            self.__distribution,
        )
        self.__acquisition.set_acquisition_algorithm(
            acquisition_algorithm, **acquisition_options
        )
        self.__regression_file_path = regression_file_path

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

            output_data = self.__problem.evaluate_functions(
                input_data, normalize=False
            )[0]
            extra_learning_set = IODataset()
            distribution = self.__distribution
            variable_names_to_n_components = distribution.algo.sizes
            extra_learning_set.add_group(
                group_name=IODataset.INPUT_GROUP,
                data=input_data[newaxis],
                variable_names=distribution.input_names,
                variable_names_to_n_components=variable_names_to_n_components,
            )
            output_names = distribution.output_names
            extra_learning_set.add_group(
                group_name=IODataset.OUTPUT_GROUP,
                data=hstack([output_data[output_name] for output_name in output_names])[
                    newaxis
                ],
                variable_names=output_names,
                variable_names_to_n_components=variable_names_to_n_components,
            )
            self.__dataset = concat(
                [distribution.algo.learning_set, extra_learning_set],
                ignore_index=True,
            )
            self.__dataset = self.__dataset.map(lambda x: x.real)
            distribution.change_learning_set(self.__dataset)
            self.__acquisition.update_problem()

            if self.__regression_file_path:
                with Path(self.__regression_file_path).open("wb") as file:
                    pickle.dump(distribution.algo, file)

        return message
