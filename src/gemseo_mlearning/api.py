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
"""Some useful functions for machine learning."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo import create_scenario
from gemseo.algos.doe.doe_library import DOELibrary
from gemseo.algos.doe.doe_library import DOELibraryOptionType
from gemseo.core.scenario import Scenario

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.datasets.dataset import Dataset


def sample_discipline(
    discipline: MDODiscipline,
    input_space: DesignSpace,
    output_names: str | Iterable[str],
    algo_name: str,
    n_samples: int,
    name: str | None = None,
    **algo_options: Any,
) -> Dataset:
    """Sample a discipline.

    Args:
        discipline: The discipline to be sampled.
        input_space: The input space on which to sample the discipline.
        output_names: The names of the outputs of interest.
        algo_name: The name of the DOE algorithm.
        n_samples: The number of samples.
        name: The name of the returned dataset.
            If ``None``, use the name of the discipline.
        **algo_options: The options of the DOE algorithm.

    Returns:
        The input-output samples of the disciplines.
    """
    return sample_disciplines(
        [discipline],
        "DisciplinaryOpt",
        input_space,
        output_names,
        algo_name,
        n_samples,
        name or discipline.name,
        **algo_options,
    )


def sample_disciplines(
    disciplines: Sequence[MDODiscipline],
    formulation: str,
    input_space: DesignSpace,
    output_names: str | Iterable[str],
    algo_name: str,
    n_samples: int,
    name: str | None = None,
    formulation_options: Mapping[str, Any] | None = None,
    **algo_options: DOELibraryOptionType,
) -> Dataset:
    """Sample several disciplines based on an MDO formulation.

    Args:
        disciplines: The disciplines to be sampled.
        formulation: The name of the MDO formulation.
        input_space: The input space on which to sample the discipline.
        output_names: The names of the outputs of interest.
        algo_name: The name of the DOE algorithm.
        n_samples: The number of samples.
        name: The name of the returned dataset.
            If ``None``, use the name of the discipline.
        formulation_options: The options of the MDO formulation.
            If ``None``, use the default ones.
        **algo_options: The options of the DOE algorithm.

    Returns:
        The input-output samples of the disciplines.
    """
    if isinstance(output_names, str):
        output_names = [output_names]

    formulation_options = formulation_options or {}
    output_names_iterator = iter(output_names)
    scenario = create_scenario(
        disciplines,
        formulation,
        next(output_names_iterator),
        input_space,
        scenario_type="DOE",
        **formulation_options,
    )
    for output_name in output_names_iterator:
        scenario.add_observable(output_name)
    scenario.execute({
        Scenario.ALGO: algo_name,
        DOELibrary.N_SAMPLES: n_samples,
        Scenario.ALGO_OPTIONS: algo_options,
    })

    return scenario.formulation.opt_problem.to_dataset(name=name, opt_naming=False)
