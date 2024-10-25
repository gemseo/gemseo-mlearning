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
from gemseo.algos.design_space import DesignSpace
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
from numpy import tile
from pandas import concat

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion_family import (  # noqa: E501
    AcquisitionCriterionFamilyFactory,
)
from gemseo_mlearning.active_learning.distributions import KrigingDistribution
from gemseo_mlearning.active_learning.distributions import RegressorDistribution
from gemseo_mlearning.active_learning.visualization.acquisition_view import (
    AcquisitionView,
)
from gemseo_mlearning.active_learning.visualization.qoi_history_view import (
    QOIHistoryView,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.algos.base_driver_library import BaseDriverLibrary
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.post.dataset.lines import Lines
    from matplotlib.figure import Figure

    from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
        BaseAcquisitionCriterion,
    )
    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )

LOGGER = logging.getLogger(__name__)


class ActiveLearningAlgo:
    """An active learning algorithm using a regressor and acquisition criteria."""

    __acquisition_algo: BaseDriverLibrary
    """The algorithm to find the new training point(s)."""

    __acquisition_algo_settings: dict[str, Any]
    """The settings of the algorithm to find the new training points()."""

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

    __default_algo_name: ClassVar[str] = "MultiStart"
    """The name of the default algorithm to find the new training point(s).

    Typically a DoE or an optimizer.
    """

    __default_algo_settings: ClassVar[dict[str, Any]] = {
        "max_iter": 200,
        "n_start": 20,
        "opt_algo_name": "SLSQP",
    }
    """The names and values of the default algorithm settings."""

    __distribution: BaseRegressorDistribution
    """The distribution of the machine learning algorithm."""

    __n_initial_samples: int
    """The number of initial samples."""

    __batch_size: int
    """The number of points to be acquired in parallel.

    If `1`, acquire points sequentially.
    """

    __n_evaluations_history: list[int]
    """The history of the number of evaluations."""

    __qoi_history: list[float]
    """The history of the quantity of interest."""

    def __init__(
        self,
        criterion_family_name: str,
        input_space: DesignSpace,
        regressor: BaseRegressor | BaseRegressorDistribution,
        criterion_name: str = "",
        batch_size: int = 1,
        mc_size: int = 10_000,
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
            batch_size: The number of points to be acquired in parallel;
                if `1`, acquire points sequentially.
            mc_size: The sample size to estimate the acquisition criteria in parallel.
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
        self.__criterion_family_name = criterion_family_name
        criterion_family = family_factory.get_class(criterion_family_name)
        criterion_factory = criterion_family.ACQUISITION_CRITERION_FACTORY()
        self.__acquisition_criterion = criterion_factory.create(
            criterion_name,
            distribution,
            batch_size=batch_size,
            mc_size=mc_size,
            **criterion_arguments,
        )
        # Create the optimization space
        # that is different from the input space
        # when acquiring points in parallel.
        optimization_space = DesignSpace()
        lower_bound = tile(input_space.get_lower_bounds(), batch_size)
        upper_bound = tile(input_space.get_upper_bounds(), batch_size)
        input_space.initialize_missing_current_values()
        value = tile(input_space.get_current_value(), batch_size)
        optimization_space.add_variable(
            "x",
            size=int(len(lower_bound)),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            value=value,
        )
        problem = self.__acquisition_problem = OptimizationProblem(optimization_space)
        problem.objective = self.__acquisition_criterion
        if not problem.objective.has_jac:
            problem.differentiation_method = (
                OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES
            )
        if problem.objective.MAXIMIZE:
            problem.minimize_objective = False

        # Initialize acquisition algorithm.
        self.set_acquisition_algorithm(
            self.__default_algo_name, **self.__default_algo_settings
        )

        # Miscellaneous.
        self.__database = Database()
        self.__n_initial_samples = len(distribution.algo.learning_set)
        self.__distribution = distribution
        self.__input_space = input_space
        self.__batch_size = batch_size

        # Create the acquisition view.
        if distribution.algo.input_dimension == 2 and batch_size == 1:
            self.__acquisition_view = AcquisitionView(self)
        else:
            self.__acquisition_view = None

        self.__n_evaluations_history = []
        self.__qoi_history = []

    @property
    def qoi(self) -> float | None:
        """The quantity of interest (QOI).

        When there is no quantity of interest associated with the acquisition criterion,
        this attribute is `None`.
        """
        return (
            None if self.__acquisition_criterion.qoi is None else self.__qoi_history[-1]
        )

    @property
    def qoi_history(self) -> tuple[list[int], list[float]]:
        """The history of the quantity of interest (QOI) when it exists.

        The second term represents this history while the first one represents the
        history of the number of model evaluations corresponding to these QOI
        estimations.

        When there is no quantity of interest associated with the acquisition criterion,
        these lists are empty.
        """
        return self.__n_evaluations_history, self.__qoi_history

    @property
    def acquisition_criterion(self) -> BaseAcquisitionCriterion:
        """The acquisition criterion."""
        return self.__acquisition_criterion

    @property
    def acquisition_criterion_family_name(self) -> str:
        """The name of the acquisition criterion family."""
        return self.__criterion_family_name

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

    @property
    def batch_size(self) -> int:
        """The number of points to be acquired in parallel."""
        return self.__batch_size

    def set_acquisition_algorithm(self, algo_name: str, **settings: Any) -> None:
        """Set sampling or optimization algorithm.

        Args:
            algo_name: The name of a DOE or optimization algorithm
                to find the learning point(s).
            **settings: The values of some algorithm settings.
        """
        factory = DOELibraryFactory()
        if not factory.is_available(algo_name):
            factory = OptimizationLibraryFactory()

        self.__acquisition_algo = factory.create(algo_name)
        self.__acquisition_algo_settings = settings

    def find_next_point(
        self,
        as_dict: bool = False,
    ) -> DataType:
        """Find the next `batch_size` learning point(s).

        Args:
            as_dict: Whether to return the input data split by input names.
                Otherwise, return a unique array.
                In both cases,
                the arrays will be shaped as ``(batch_size, input_dimension)``.

        Returns:
            The next `batch_size` learning point(s).
        """
        with LoggingContext(logging.getLogger("gemseo")):
            input_data = self.__acquisition_algo.execute(
                self.__acquisition_problem, **self.__acquisition_algo_settings
            ).x_opt
            input_data = input_data.reshape(self.__batch_size, -1)
        if as_dict:
            return self.__acquisition_problem.design_space.convert_array_to_dict(
                input_data
            )

        return input_data

    def acquire_new_points(
        self,
        discipline: Discipline,
        n_samples: int = 1,
        show: bool = False,
        file_path: str | Path = "",
    ) -> tuple[Database, OptimizationProblem]:
        """Update the machine learning algorithm by learning new samples.

        This method acquires new learning input-output samples
        and trains the machine learning algorithm
        with the resulting enriched learning set.
        The effective number of points will be the largest integer multiple
        of batch_size and less than or equal to n_samples.

        Args:
            discipline: The discipline computing the reference output data
                from the input data provided by the acquisition process.
            n_samples: The number of samples to update the machine learning algorithm.
                It should be a multiple of batch_size.
            show: Whether to display intermediate results
                Only when the input space dimension is 2.
            file_path: The file path to save the plots of the intermediate results.
                If empty, do not save them.
                Only when the input space dimension is 2.

        Returns:
            The concatenation of the optimization histories
            related to the different points
            and the last acquisition problem.

        Raises:
            ValueError: When the input space dimension is not 2.
        """
        plot = show or file_path
        if plot:
            self.__check_acquisition_view()

        self.__n_evaluations_history.append(self.__n_initial_samples)
        self.__qoi_history.append(self.__acquisition_criterion.qoi)
        total_n_samples = self.__n_initial_samples
        n_batches = int(n_samples / self.__batch_size)
        LOGGER.info("Acquiring %s points in batches of %s", n_samples, self.batch_size)
        with OneLineLogging(TQDM_LOGGER):
            for batch_id in CustomTqdmProgressBar(range(1, n_batches + 1)):
                array_input_data = self.find_next_point()
                if plot:
                    self.__acquisition_view.draw(
                        discipline=discipline,
                        new_point=array_input_data[0],
                        show=show,
                        file_path=file_path,
                    )

                for inputs, outputs in self.__acquisition_problem.database.items():
                    self.__database.store(
                        array([batch_id, *inputs.unwrap().tolist()]), outputs
                    )

                for points in range(self.__batch_size):
                    input_data = self.__input_space.convert_array_to_dict(
                        array_input_data[points, :]
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
                self.__qoi_history.append(self.__acquisition_criterion.qoi)
                total_n_samples += self.__batch_size
                self.__n_evaluations_history.append(total_n_samples)

        return self.__database, self.__acquisition_problem

    def __check_acquisition_view(self) -> None:
        """Check that the acquisition view can be used.

        Raises:
            ValueError: When the input space dimension is not 2.
        """
        if not self.__acquisition_view:
            msg = (
                "Plotting intermediate results "
                "requires an input space dimension equal to 2."
            )
            raise ValueError(msg)

    def plot_acquisition_view(
        self,
        discipline: Discipline | None = None,
        filled: bool = True,
        n_test: int = 30,
        show: bool = True,
        file_path: str | Path = "",
    ) -> Figure:
        """Draw the points acquired through active learning.

        This visualization includes four surface plots
        representing variables of interest in function of the two inputs:

        - (top left) the surface plot of the discipline,
        - (top right) the surface plot of the acquisition criterion,
        - (bottom left) the surface plot of the regressor,
        - (bottom right) the standard deviation of the regressor.

        The $N$ data used to initialize the regressor are represented by small dots,
        the point optimizing the acquisition criterion is represented by a big dot
        (this will be the next point to learn)
        and the points added by the active learning procedure are represented
        by plus signs and labeled by their position in the learning dataset.

        Args:
            discipline: The discipline for which `n_test**2` evaluations will be done.
                If `None`,
                do not plot the discipline, i.e. the first subplot.
                In particular,
                if the discipline is costly,
                it is better to leave this argument to `None`.
            n_test: The number of points per dimension.
            filled: Whether to plot filled contours.
                Otherwise, plot contour lines.
            show: Whether to display the view.
            file_path: The file path to save the view.
                If empty, do not save it.

        Returns:
            The acquisition view.

        Raises:
            ValueError: When the input space dimension is not 2.
        """
        self.__check_acquisition_view()
        return self.__acquisition_view.draw(
            discipline=discipline,
            filled=filled,
            n_test=n_test,
            show=show,
            file_path=file_path,
        )

    def plot_qoi_history(
        self,
        show: bool = True,
        file_path: str | Path = "",
        label: str = "Quantity of interest",
        add_markers: bool = True,
        **options: Any,
    ) -> Lines:
        """Plot the history of the quantity of interest.

        Args:
            show: Whether to display the view.
            file_path: The file path to save the view.
                If empty, do not save it.
            label: The label for the QOI.
            add_markers: Whether to add markers.
            **options: The options to create the
                [Lines][gemseo.post.dataset.lines.Lines] object.

        Returns:
            The history of the quantity of interest.
        """
        return QOIHistoryView(self).draw(
            show=show,
            file_path=file_path,
            label=label,
            add_markers=add_markers,
            **options,
        )

    def update_problem(self) -> None:
        """Update the acquisition problem."""
        self.__acquisition_problem.reset(preprocessing=False)
        self.__acquisition_criterion.update()
