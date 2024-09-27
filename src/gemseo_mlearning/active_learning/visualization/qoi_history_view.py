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
"""History view."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.datasets.io_dataset import IODataset
from gemseo.post.dataset.lines import Lines
from numpy import array
from numpy import newaxis

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo


class QOIHistoryView:
    """View of the history of the quantity of interest (QOI)."""

    __algo: ActiveLearningAlgo
    """The active learning algorithm."""

    def __init__(self, active_learning_algo: ActiveLearningAlgo) -> None:
        """
        Args:
            active_learning_algo: The active learning algorithm.

        Raises:
            ValueError: When there is no quantity of interest
                associated with the acquisition criterion.
        """  # noqa: D205, D212
        if active_learning_algo.acquisition_criterion.qoi is None:
            msg = (
                "There is no quantity of interest "
                "associated with the acquisition criterion "
                f"{active_learning_algo.acquisition_criterion.__class__.__name__}."
            )
            raise ValueError(msg)

        self.__algo = active_learning_algo

    def draw(
        self,
        show: bool = True,
        file_path: str | Path = "",
        label: str = "",
        add_markers: bool = True,
        **options: Any,
    ) -> Lines:
        """Draw the QOI history.

        Args:
            show: Whether to display the view.
            file_path: The file path to save the view.
                If empty, do not save it.
            label: The label for the QOI.
                If empty, use the name of the acquisition criterion family.
            add_markers: Whether to add markers.
            **options: The options to create the
                [Lines][gemseo.post.dataset.lines.Lines] object.

        Returns:
            A view of the QOI history.
        """
        if not label:
            label = self.__algo.acquisition_criterion_family_name

        x_label = "Number of evaluations"
        n_evaluations_history, qoi_history = self.__algo.qoi_history
        qoi_history = [qoi[0] for qoi in qoi_history]
        dataset = IODataset()
        dataset.add_variable(x_label, array(n_evaluations_history)[:, newaxis])
        dataset.add_variable(label, array(qoi_history)[:, newaxis])
        lines = Lines(
            dataset,
            variables=[label],
            abscissa_variable=x_label,
            add_markers=add_markers,
            **options,
        )
        lines.marker = "."
        lines.execute(show=show, save=file_path != "", file_path=file_path)
        return lines
