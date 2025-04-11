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

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.hashable_ndarray import HashableNdarray
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.factory import RegressorFactory
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.logging_tools import LoggingContext
from numpy import hstack
from numpy import newaxis
from pandas import concat

from gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum import (
    Minimum,
)
from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.base_driver_library import DriverLibrarySettingType
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.mlearning.core.algos.ml_algo import MLAlgoParameterType


class SurrogateBasedOptimizer:
    """An optimizer based on surrogate models."""

    __STOP_BECAUSE_ALREADY_KNOWN = "The acquired input data is already known."
    __STOP_BECAUSE_MAX_ACQUISITIONS = "All the data acquisitions have been made."

    __active_learning_algo: ActiveLearningAlgo
    """The active learning algorithm to acquire new samples to learn."""

    __dataset: IODataset
    """The original learning dataset enriched by new samples."""

    __initial_input_samples: tuple[HashableNdarray, ...]
    """The initial input samples, if any."""

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
        doe_algorithm: str = "OT_OPT_LHS",
        doe_settings: Mapping[str, DriverLibrarySettingType] = READ_ONLY_EMPTY_DICT,
        regression_algorithm: str | BaseRegressor = OTGaussianProcessRegressor.__name__,
        regression_settings: Mapping[str, MLAlgoParameterType] = READ_ONLY_EMPTY_DICT,
        regression_file_path: str | Path = "",
        **acquisition_settings: DriverLibrarySettingType,
    ) -> None:
        """
        Args:
            acquisition_algorithm: The name of the algorithm to optimize the data
                acquisition criterion.
                N.B. this algorithm must handle integers if some of the optimization
                variables are integers.
            problem: The optimization problem.
            doe_size: Either the size of the initial DOE
                or 0 if the size is inferred from doe_settings.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            doe_algorithm: The name of the algorithm for the initial sampling.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            doe_settings: The settings of the algorithm for the initial sampling.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            regression_algorithm: Either the name of the regression algorithm
                approximating the objective function over the design space
                or the regression algorithm itself.
            regression_settings: The settings of the regression algorithm.
                If transformer is missing,
                use :attr:`.BaseMLRegressionAlgo.DEFAULT_TRANSFORMER`.
                This argument is ignored
                when regression_algorithm is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            regression_file_path: The path to the file to save the regression model.
                If empty, do not save the regression model.
            **acquisition_settings: The settings of the algorithm to optimize
                the data acquisition criterion.
        """  # noqa: D205, D212, D415
        self.__problem = problem
        database = problem.database
        self.__initial_input_samples = tuple(database.keys())
        if isinstance(regression_algorithm, BaseRegressor):
            self.__dataset = regression_algorithm.learning_set
        else:
            # Store max_iter as it will be overwritten by DOELibrary
            max_iter = problem.evaluation_counter.maximum
            settings = dict(doe_settings)
            if doe_size > 0 and "n_samples" not in settings:
                settings["n_samples"] = doe_size

            # Store the listeners as they will be cleared by DOELibrary.
            new_iter_listeners, store_listeners = database.clear_listeners()
            with LoggingContext(logging.getLogger("gemseo")):
                DOELibraryFactory().execute(
                    problem, algo_name=doe_algorithm, **settings
                )

            for listener in new_iter_listeners:
                database.add_new_iter_listener(listener)

            for listener in store_listeners:
                database.add_store_listener(listener)

            self.__dataset = problem.to_dataset(opt_naming=False)
            if self.__initial_input_samples:
                self.__dataset = self.__dataset[len(self.__initial_input_samples) :]

            regression_settings_ = {"transformer": {"inputs": "MinMaxScaler"}}
            regression_settings_.update(dict(regression_settings))
            regression_algorithm = RegressorFactory().create(
                regression_algorithm,
                self.__dataset,
                **regression_settings_,
            )
            # Add the first iteration to the current_iter reset by DOELibrary.
            problem.evaluation_counter.current += 1
            # And restore max_iter.
            problem.evaluation_counter.maximum = max_iter

        self.__active_learning_algo = ActiveLearningAlgo(
            Minimum.__name__, problem.design_space, regression_algorithm
        )
        if acquisition_algorithm:
            self.__active_learning_algo.set_acquisition_algorithm(
                acquisition_algorithm, **acquisition_settings
            )
        self.__regression_file_path = regression_file_path

    def execute(self, number_of_acquisitions: int) -> str:
        """Execute the surrogate-based optimization.

        Args:
            number_of_acquisitions: The number of learning points to be acquired.

        Returns:
            The termination message.
        """
        regressor_distribution = self.__active_learning_algo.regressor_distribution
        regressor_distribution.learn()
        message = self.__STOP_BECAUSE_MAX_ACQUISITIONS
        for _ in range(number_of_acquisitions):
            input_data = self.__active_learning_algo.find_next_point()[0]
            hashed_input_data = HashableNdarray(input_data)
            if hashed_input_data in self.__problem.database and (
                hashed_input_data not in self.__initial_input_samples
            ):
                message = self.__STOP_BECAUSE_ALREADY_KNOWN
                break

            output_data = self.__problem.evaluate_functions(
                design_vector=input_data, design_vector_is_normalized=False
            )[0]
            extra_learning_set = IODataset()
            variable_names_to_n_components = regressor_distribution.algo.sizes
            extra_learning_set.add_group(
                group_name=IODataset.INPUT_GROUP,
                data=input_data[newaxis],
                variable_names=regressor_distribution.input_names,
                variable_names_to_n_components=variable_names_to_n_components,
            )
            output_names = regressor_distribution.output_names
            extra_learning_set.add_group(
                group_name=IODataset.OUTPUT_GROUP,
                data=hstack([output_data[output_name] for output_name in output_names])[
                    newaxis
                ],
                variable_names=output_names,
                variable_names_to_n_components=variable_names_to_n_components,
            )
            self.__dataset = concat(
                [regressor_distribution.algo.learning_set, extra_learning_set],
                ignore_index=True,
            )
            self.__dataset = self.__dataset.map(lambda x: x.real)
            regressor_distribution.change_learning_set(self.__dataset)
            self.__active_learning_algo.update_problem()

            if self.__regression_file_path:
                with Path(self.__regression_file_path).open("wb") as file:
                    pickle.dump(regressor_distribution.algo, file)

        return message
