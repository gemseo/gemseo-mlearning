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

from gemseo.dataset import IODataset
from gemseo.doe.factory import DOELibraryFactory
from gemseo.machine_learning.regression.core.base_regressor import BaseRegressor
from gemseo.machine_learning.regression.model.factory import regressor_factory
from gemseo.machine_learning.regression.model.ot_gpr_settings import (
    OTGaussianProcessRegressor_Settings,
)
from gemseo.util.constant import read_only_empty_dict
from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.logging import LoggingContext
from numpy import hstack
from numpy import newaxis
from pandas import concat

from gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum import (
    Minimum,
)
from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.algorithm.base_driver_library import DriverLibrarySettingType
    from gemseo.machine_learning.regression.core.base_regressor_settings import (
        BaseRegressorSettings,
    )
    from gemseo.optimization import OptimizationProblem


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
        doe_settings: Mapping[str, DriverLibrarySettingType] = read_only_empty_dict,
        regressor: BaseRegressorSettings | BaseRegressor | None = None,
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
                when regressor is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            doe_algorithm: The name of the algorithm for the initial sampling.
                This argument is ignored
                when regressor is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            doe_settings: The settings of the algorithm for the initial sampling.
                This argument is ignored
                when regressor is a
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            regressor: Either regressor settings or a regressor.
                If `None`, use the default OpenTURNS-based Gaussian process regressor.
            regression_file_path: The path to the file to save the regression model.
                If empty, do not save the regression model.
            **acquisition_settings: The settings of the algorithm to optimize
                the data acquisition criterion.
        """  # noqa: D205, D212, D415
        if regressor is None:
            regressor = OTGaussianProcessRegressor_Settings()
        self.__problem = problem
        database = problem.database
        self.__initial_input_samples = tuple(database.keys())
        if isinstance(regressor, BaseRegressor):
            self.__dataset = regressor.learning_set
        else:
            # Store max_iter as it will be overwritten by DOELibrary
            max_iter = problem.evaluation_counter.maximum
            settings = dict(doe_settings)
            if doe_size > 0 and "n_samples" not in settings:
                settings["n_samples"] = doe_size

            # Store the listeners as they will be cleared by DOELibrary.
            new_iter_listeners, store_listeners = database.clear_listeners()
            with LoggingContext(logging.getLogger("gemseo")):
                # TODO(bump-gemseo): cannot transform: the Settings class is named by doe_algorithm, which is not a string literal  # noqa: E501
                # TODO(bump-gemseo): **kwargs may contain: settings_model -> settings  # noqa: E501
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

            if "transformer" not in regressor.model_fields_set:
                regressor.transformer = {"inputs": "MinMaxScaler"}
            # TODO(bump-gemseo): cannot transform: the type of regressor could not be inferred; if it is an instance of BaseSettings, read it with the target_class_name property of a settings instance; in a settings class, remove the assignment, as the value is now derived from the name of the class (X_Settings targets X)  # noqa: E501
            regressor = regressor_factory.create(
                regressor._TARGET_CLASS_NAME, self.__dataset, settings=regressor
            )
            # Add the first iteration to the current_iter reset by DOELibrary.
            problem.evaluation_counter.current += 1
            # And restore max_iter.
            problem.evaluation_counter.maximum = max_iter

        self.__active_learning_algo = ActiveLearningAlgo(
            Minimum.__name__, problem.design_space, regressor
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
                input_value=input_data, input_value_is_normalized=False
            )[0]
            extra_learning_set = IODataset()
            variable_name_to_n_components = regressor_distribution.regressor.sizes
            extra_learning_set.add_group(
                group_name=IODataset.input_group,
                data=input_data[newaxis],
                variable_names=regressor_distribution.input_names,
                variable_name_to_n_components=variable_name_to_n_components,
            )
            output_names = regressor_distribution.output_names
            extra_learning_set.add_group(
                group_name=IODataset.output_group,
                data=hstack([output_data[output_name] for output_name in output_names])[
                    newaxis
                ],
                variable_names=output_names,
                variable_name_to_n_components=variable_name_to_n_components,
            )
            self.__dataset = concat(
                [regressor_distribution.regressor.learning_set, extra_learning_set],
                ignore_index=True,
            )
            self.__dataset = self.__dataset.map(lambda x: x.real)
            regressor_distribution.change_learning_set(self.__dataset)
            self.__active_learning_algo.update_problem()

            if self.__regression_file_path:
                with Path(self.__regression_file_path).open("wb") as file:
                    pickle.dump(regressor_distribution.regressor, file)

        return message
