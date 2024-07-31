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
"""An acquisition plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.utils.matplotlib_figure import save_show_figure
from matplotlib import pyplot as plt
from numpy import array
from numpy import linspace
from numpy import meshgrid
from numpy import zeros

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.core.discipline import MDODiscipline
    from gemseo.typing import RealArray
    from matplotlib.figure import Figure

    from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo


class AcquisitionView:
    """View of points acquired during active learning.

    This visualization tool only works with a 2-dimensional input space.
    """

    __algo: ActiveLearningAlgo
    """The active learning algorithm."""

    __input_dimension: int
    """The input dimension."""

    def __init__(self, active_learning_algo: ActiveLearningAlgo) -> None:
        """
        Args:
            active_learning_algo: The active learning algorithm.
        """  # noqa: D205, D212
        self.__algo = active_learning_algo
        self.__input_dimension = (
            active_learning_algo.regressor_distribution.algo.input_dimension
        )

    def draw(
        self,
        new_point: RealArray | None = None,
        discipline: MDODiscipline | None = None,
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
            new_point: The new point to be acquired.
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
        """
        # Create grid.
        input_space = self.__algo.input_space
        lower_bounds = input_space.get_lower_bounds()
        upper_bounds = input_space.get_upper_bounds()
        test_x1 = linspace(lower_bounds[0], upper_bounds[0], n_test)
        test_x2 = linspace(lower_bounds[1], upper_bounds[1], n_test)
        grid = array(meshgrid(test_x1, test_x2)).T.reshape(-1, 2)

        # Generate data.
        distribution = self.__algo.regressor_distribution
        final_dataset = distribution.algo.learning_set
        #    The learning input samples.
        points = final_dataset.input_dataset.to_numpy()
        points_x = points[:, 0]
        points_y = points[:, 1]
        #    The predictions, the standard deviations and the criterion values.
        predictions = distribution.predict(grid).reshape((n_test, n_test)).T
        std = distribution.compute_standard_deviation(grid).reshape((n_test, n_test)).T
        acquisition_criterion = self.__algo.acquisition_criterion.original.func
        criterion_values = acquisition_criterion(grid).reshape((n_test, n_test)).T
        x_name, y_name = final_dataset.input_names
        output_name = final_dataset.output_names[0]
        #    The observations if the discipline is available.
        observations = None
        if discipline is not None:
            observations = zeros((n_test, n_test))
            for i in range(n_test):
                for j in range(n_test):
                    xij = array([test_x1[j], test_x2[i]])
                    input_data = {x_name: array([xij[0]]), y_name: array([xij[1]])}
                    observations[i, j] = discipline.execute(input_data)[output_name][0]

        # Create figure and sub-figures.
        fig, axes = plt.subplots(2, 2)
        titles = [
            ["Discipline", self.__algo.acquisition_criterion.__class__.__name__],
            ["Surrogate", "Standard deviation"],
        ]
        data = [[observations, criterion_values], [predictions, std]]
        cf = []
        color = "white" if filled else "black"
        n_initial_samples = self.__algo.n_initial_samples
        contour_method = "contourf" if filled else "contour"
        for i in range(2):
            for j in range(2):
                if (i, j) == (0, 0) and discipline is None:
                    continue

                ax = axes[i, j]
                cf.append(getattr(ax, contour_method)(test_x1, test_x2, data[i][j]))
                for x, y in zip(
                    points_x[:n_initial_samples],
                    points_y[:n_initial_samples],
                ):
                    ax.plot(x, y, ".", ms=2, color=color)

                for index, (x, y) in enumerate(
                    zip(
                        points_x[n_initial_samples:],
                        points_y[n_initial_samples:],
                    )
                ):
                    ax.plot(x, y, "+", color=color)
                    ax.annotate(
                        str(n_initial_samples + 1 + index),
                        (0.05 + x, 0.05 + y),
                        color=color,
                    )

                if new_point is not None:
                    ax.plot(*new_point, "o", color=color)

                ax.set_title(titles[i][j])

        if discipline is None:
            fig.delaxes(axes.flatten()[0])
            fig.colorbar(cf[0], ax=axes[0, 1])
            axes[0, 1].set_xticks([])
            fig.colorbar(cf[1], ax=axes[1, 0])
            fig.colorbar(cf[2], ax=axes[1, 1])
            axes[1, 1].set_yticks([])
        else:
            fig.colorbar(cf[0], ax=axes[:, 0])
            axes[0, 0].set_xticks([])
            fig.colorbar(cf[1], ax=axes[0, 1])
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
            fig.colorbar(cf[3], ax=axes[1, 1])
            axes[1, 1].set_yticks([])

        save_show_figure(fig, show, file_path)
        return fig
