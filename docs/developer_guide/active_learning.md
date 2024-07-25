<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# Active learning

This section describes the design of the [active_learning][gemseo_mlearning.active_learning] subpackage.

!!! info

    Open the [user guide](../user_guide/active_learning.md) for general information, *e.g.* concepts, API, examples, etc.

## Tree structure

```tree
gemseo_mlearning
  active_learning # Subpackage for active learning (AL)
    active_learning_algo.py # Class to set and solve an AL problem
    acquisition_criteria # Acquisition criteria (ACs)
      exploration # ACs to improve the regressor
        base_exploration.py # Base class for these ACs
        distance.py # AC (distance to the learning set)
        exploration.py # Specific AC family
        factory.py # Factory for these ACs
        standard_deviation.py # An AC (regressor's standard deviation)
        variance.py # AC (regressor's variance)
      level_set/ # ACs to approximate a level set
      maximum/ # ACs to approximate the global maximum
      minimum/ # ACs to approximate the global minimum
      quantile/ # ACs to approximate a quantile
      base_acquisition_criterion.py # Base class for ACs
      base_acquisition_criterion_family.py # Base class for AC families
      base_factory.py # Base class for AC
    distributions # Regressor distributions (RD)
      base_regressor_distribution.py # The base class for RDs
      kriging_distribution.py # The RD for Kriging regressors
      regressor_distribution.py # The RD for any regressor
    visualization # Visualization tools
      acquisition_view.py # Plot the acquisition process (in 2D only)
```

## Class diagram

``` mermaid
classDiagram

    ActiveLearning *-- OptimizationProblem
    ActiveLearning o-- DesignSpace
    ActiveLearning --> AcquisitionCriterionFamilyFactory: criterion_family_name
    ActiveLearning --> BaseAcquisitionCriterionFactory: criterion_name \n criterion_options
    ActiveLearning --> BaseRegressorDistribution: regressor
    BaseDriverLibrary --* ActiveLearning
    BaseDOELibrary --* BaseDriverLibrary
    BaseOptimizationLibrary --* BaseDriverLibrary
    Database --* ActiveLearning
    AcquisitionView --* ActiveLearning

    OptimizationProblem o-- BaseAcquisitionCriterion

    AcquisitionCriterionFamilyFactory --> BaseAcquisitionCriterionFamily
    BaseAcquisitionCriterionFamily --> BaseAcquisitionCriterionFactory
    BaseAcquisitionCriterionFactory --> BaseAcquisitionCriterion

    BaseAcquisitionCriterion --|> MDOFunction
    BaseAcquisitionCriterion o-- BaseRegressorDistribution

    <<abstract>> BaseAcquisitionCriterion
    <<abstract>> BaseAcquisitionCriterionFactory
    <<abstract>> BaseAcquisitionCriterionFamily
    <<abstract>> BaseRegressorDistribution

    class ActiveLearning {
        +default_algo_name
        +default_doe_options
        +default_opt_options
        +acquisition_criterion
        +n_initial_samples
        +regressor
        +regressor_distribution
        +set_acquisition_algorithm()
        +find_nex_point()
        +acquire_new_points()
        +update_problem()
    }

    class AcquisitionCriterionFamilyFactory {
        +create()
    }

    class BaseAcquisitionCriterionFamily {
        +ACQUISITION_CRITERION_FACTORY
    }

    class BaseAcquisitionCriterionFactory {
        #DEFAULT_CLASS_NAME
    }
```

## How to...

`... create a new family of acquisition criteria?`

:   1. Derive an abstract class `BaseNewAcquisitionCriterion` from
       [BaseAcquisitionCriterion][gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion.BaseAcquisitionCriterion].
    1. Derive `FirstAcquisitionCriterion`, `SecondAcquisitionCriterion`, ... from `BaseNewAcquisitionCriterion`
       in modules located in `root_package_name.subpackage_name.package_name`.
    1. Derive `NewAcquisitionCriterionFactory` from
       [BaseAcquisitionCriterionFactory][gemseo_mlearning.active_learning.acquisition_criteria.base_factory.BaseAcquisitionCriterionFactory]
       and set the class attributes:
       ```python
       _CLASS = BaseNewAcquisitionCriterion
       _DEFAULT_CLASS_NAME = "FirstAcquisitionCriterion"
       _MODULE_NAMES = ("root_package.subpackage_name.package_name",)
       ```
    1. Derive `NewAcquisitionCriterionFamily` from
       [BaseAcquisitionCriterionFamily][gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion_family.BaseAcquisitionCriterionFamily]
       and set the class attribute `ACQUISITION_CRITERION_FACTORY = NewAcquisitionCriterionFactory`.

    Now,
    the user can instantiate
    the [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]

    - with `#!python criterion_family_name="NewAcquisitionCriterionFamily"`
      to use this family of acquisition criteria,
    - with `#!python criterion_family_name="NewAcquisitionCriterionFamily"`
      and `#!python criterion_name="SecondAcquisitionCriterion"`
      to use a specific acquisition criterion from this family.
