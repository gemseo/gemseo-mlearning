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
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.core.base_factory import BaseFactory
from gemseo.mlearning.core.algos.ml_algo_settings import BaseMLAlgoSettings

import gemseo_mlearning.settings.mlearning as mlearning
import gemseo_mlearning.settings.opt as opt

if TYPE_CHECKING:
    from gemseo.algos.base_algorithm_settings import BaseAlgorithmSettings


def get_setting_classes(
    BaseSettings: type[BaseAlgorithmSettings],  # noqa: N803
    package_name: str,
    module_,
) -> list[str]:
    """Return the settings classes given a type of algorithms.

    Args:
        BaseSettings: The base class specific to the type of algorithms.
        package_name: The name of the package.
        module_: The module of settings.

    Returns:
        The settings classes.
    """

    class SettingsFactory(BaseFactory):
        _CLASS = BaseSettings
        _PACKAGE_NAMES = (package_name,)

        @property
        def classes(self) -> list[str]:
            return [
                self.get_class(name)
                for name in super().class_names
                if not name.startswith("Base")
            ]

    for cls in SettingsFactory().classes:
        if cls.__module__.startswith("gemseo_mlearning"):
            yield module_, cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseOptimizerSettings, "gemseo_mlearning.algos.opt", opt),
)
def test_opt_settings(module_and_cls):
    """Check aliases for optimizer settings."""
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseMLAlgoSettings, "gemseo_mlearning.regression", mlearning),
)
def test_machine_learning_settings(module_and_cls):
    """Check aliases for machine learning algorithm settings."""
    module, cls = module_and_cls
    assert cls in module.__dict__.values()
