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
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Active learning algorithm."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.algos._progress_bars.custom_tqdm_progress_bar import LOGGER as TQDM_LOGGER
from gemseo.algos._progress_bars.custom_tqdm_progress_bar import CustomTqdmProgressBar
from gemseo.algos.database import Database
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.base_random_process_regressor import (
    BaseRandomProcessRegressor,
)
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.logging_tools import OneLineLogging
from numpy import array
from numpy import hstack
from numpy import newaxis
from pandas import concat

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion_family import (  # noqa: E501
    AcquisitionCriterionFamilyFactory,
)
from gemseo_mlearning.active_learning.distributions import KrigingDistribution
from gemseo_mlearning.active_learning.distributions import RegressorDistribution
from gemseo_mlearning.active_learning.visualization.acquisition_view import (
    AcquisitionView,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
    from gemseo.core.discipline import MDODiscipline
    from gemseo.mlearning.core.algos.ml_algo import DataType

    from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
        BaseAcquisitionCriterion,
    )
    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )

LOGGER = logging.getLogger(__name__)


class ActiveLearningAlgo:
    """An active learning algorithm using a regressor and acquisition criteria."""

    __acquisition_algo: BaseOptimizationLibrary
    """The algorithm to find the new training point(s)."""

    __acquisition_algo_options: dict[str, Any]
    """The options of the algorithm to find the new training points()."""

    __acquisition_criterion: BaseAcquisitionCriterion
    """The acquisition criterion."""

    __acquisition_problem: OptimizationProblem
    """The acquisition problem."""

    __acquisition_view: AcquisitionView | None
    """View of points acquired during active learning if the input dimension is 2."""

    __database: Database
    """The concatenation of the optimization histories related to the new points."""

    __input_space: DesignSpace
    """The input space on which to look for the new learning point."""

    default_algo_name: ClassVar[str] = "NLOPT_COBYLA"
    """The name of the default algorithm to find the new training point(s).

    Typically a DoE or an optimizer.
    """

    default_doe_options: ClassVar[dict[str, Any]] = {"n_samples": 100}
    """The names and values of the default DoE options."""

    default_opt_options: ClassVar[dict[str, Any]] = {"max_iter": 100}
    """The names and values of the default optimization options."""

    __distribution: BaseRegressorDistribution
    """The distribution of the machine learning algorithm."""

    __n_initial_samples: int
    """The number of initial samples."""

    def __init__(
        self,
        criterion_family_name: str,
        input_space: DesignSpace,
        regressor: BaseRegressor | BaseRegressorDistribution,
        criterion_name: str = "",
        **criterion_arguments: Any,
    ) -> None:
        """
        Args:
            criterion_family_name: The name of a family of acquisition criteria,
                *e.g.* `"Minimum"`, `"Maximum"`, `"LevelSet"`, `"Quantile"`
                or `"Exploration"`.
            input_space: The input space on which to look for the new learning point.
            regressor: Either a regressor or a regressor distribution.
            criterion_name: The name of the acquisition criterion.
                If empty,
                use the default criterion of the family `criterion_family_name`.
            **criterion_arguments: The parameters of the acquisition criterion.

        Raises:
            NotImplementedError: When the output dimension is greater than 1.
        """  # noqa: D205 D212 D415
        # Create the regressor distribution.
        if isinstance(regressor, BaseRandomProcessRegressor):
            distribution = KrigingDistribution(regressor)
            distribution.learn()
        elif isinstance(regressor, BaseRegressor):
            distribution = RegressorDistribution(regressor)
            distribution.learn()
        else:
            distribution = regressor

        if distribution.output_dimension > 1:
            msg = "ActiveLearningAlgo works only with scalar output."
            raise NotImplementedError(msg)

        # Create the acquisition problem.
        family_factory = AcquisitionCriterionFamilyFactory()
        criterion_family = family_factory.get_class(criterion_family_name)
        criterion_factory = criterion_family.ACQUISITION_CRITERION_FACTORY()
        self.__acquisition_criterion = criterion_factory.create(
            criterion_name, distribution, **criterion_arguments
        )
        problem = self.__acquisition_problem = OptimizationProblem(input_space)
        problem.objective = self.__acquisition_criterion
        if not problem.objective.has_jac:
            problem.differentiation_method = (
                OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES
            )
        if problem.objective.MAXIMIZE:
            problem.minimize_objective = False

        # Initialize acquisition algorithm.
        optimization_factory = OptimizationLibraryFactory()
        self.__acquisition_algo = optimization_factory.create(self.default_algo_name)
        self.__acquisition_algo_options = self.default_opt_options

        # Miscellaneous.
        self.__database = Database()
        self.__n_initial_samples = len(distribution.algo.learning_set)
        self.__distribution = distribution
        self.__input_space = input_space

        # Create the acquisition view.
        if distribution.algo.input_dimension == 2:
            self.__acquisition_view = AcquisitionView(self)
        else:
            self.__acquisition_view = None

    @property
    def acquisition_criterion(self) -> BaseAcquisitionCriterion:
        """The acquisition criterion."""
        return self.__acquisition_criterion

    @property
    def regressor(self) -> BaseRegressor:
        """The regressor."""
        return self.__distribution.algo

    @property
    def regressor_distribution(self) -> BaseRegressorDistribution:
        """The regressor distribution."""
        return self.__distribution

    @property
    def n_initial_samples(self) -> int:
        """The number of initial samples."""
        return self.__n_initial_samples

    @property
    def input_space(self) -> DesignSpace:
        """The input space."""
        return self.__input_space

    def set_acquisition_algorithm(self, algo_name: str, **options: Any) -> None:
        """Set sampling or optimization algorithm.

        Args:
            algo_name: The name of a DOE or optimization algorithm
                to find the learning point(s).
            **options: The values of some algorithm options;
                use the default values for the other ones.
        """
        factory = DOELibraryFactory()
        if factory.is_available(algo_name):
            self.__acquisition_algo_options = self.default_doe_options.copy()
        else:
            factory = OptimizationLibraryFactory()
            self.__acquisition_algo_options = self.default_opt_options.copy()

        self.__acquisition_algo_options.update(options)
        self.__acquisition_algo = factory.create(algo_name)

    def find_next_point(
        self,
        as_dict: bool = False,
    ) -> DataType:
        """Find the next learning point.

        Args:
            as_dict: Whether to return the input data split by input names.
                Otherwise, return a unique array.

        Returns:
            The next learning point.
        """
        with LoggingContext(logging.getLogger("gemseo")):
            input_data = self.__acquisition_algo.execute(
                self.__acquisition_problem, **self.__acquisition_algo_options
            ).x_opt

        if as_dict:
            return self.__acquisition_problem.design_space.array_to_dict(input_data)

        return input_data

    def acquire_new_points(
        self,
        discipline: MDODiscipline,
        n_samples: int = 1,
        show: bool = False,
        file_path: str | Path = "",
    ) -> tuple[Database, OptimizationProblem]:
        """Update the machine learning algorithm by learning new samples.

        This method acquires new learning input-output samples
        and trains the machine learning algorithm
        with the resulting enriched learning set.

        Args:
            discipline: The discipline computing the reference output data
                from the input data provided by the acquisition process.
            n_samples: The number of samples to update the machine learning algorithm.
            show: Whether to display intermediate results.
            file_path: The file path to save the view.
                If empty, do not save it.

        Returns:
            The concatenation of the optimization histories
            related to the different points
            and the last acquisition problem.
        """
        LOGGER.info("Acquiring %s points", n_samples)
        with OneLineLogging(TQDM_LOGGER):
            for sample_id in CustomTqdmProgressBar(range(1, n_samples + 1)):
                array_input_data = self.find_next_point()
                if self.__acquisition_view and (show or file_path):
                    self.__acquisition_view.draw(
                        discipline=discipline,
                        new_point=array_input_data,
                        show=show,
                        file_path=file_path,
                    )

                input_data = self.__acquisition_problem.design_space.array_to_dict(
                    array_input_data
                )
                for inputs, outputs in self.__acquisition_problem.database.items():
                    self.__database.store(
                        array([sample_id, *inputs.unwrap().tolist()]), outputs
                    )

                discipline.execute(input_data)

                extra_learning_set = IODataset()
                distribution = self.__distribution
                variable_names_to_n_components = distribution.algo.sizes
                new_points = hstack(list(input_data.values()))[newaxis]
                extra_learning_set.add_group(
                    group_name=IODataset.INPUT_GROUP,
                    data=new_points,
                    variable_names=distribution.input_names,
                    variable_names_to_n_components=variable_names_to_n_components,
                )
                output_names = distribution.output_names
                output_data = discipline.get_output_data()
                extra_learning_set.add_group(
                    group_name=IODataset.OUTPUT_GROUP,
                    data=hstack(list(output_data.values()))[newaxis],
                    variable_names=output_names,
                    variable_names_to_n_components=variable_names_to_n_components,
                )
                augmented_learning_set = concat(
                    [distribution.algo.learning_set, extra_learning_set],
                    ignore_index=True,
                )

                self.__distribution.change_learning_set(augmented_learning_set)
                self.update_problem()

        return self.__database, self.__acquisition_problem

    def update_problem(self) -> None:
        """Update the acquisition problem."""
        self.__acquisition_problem.reset(preprocessing=False)
        self.__acquisition_criterion.update()
