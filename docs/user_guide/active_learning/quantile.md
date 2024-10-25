<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Quantile

[Active learning](what_active_learning_is.md) techniques are very useful
to estimate a level set,
 _i.e._ the input values for which the model output is equal to a specific value $y$.

In particular,
this specific value can be the $\alpha$-quantile for a specific $\alpha\in]0,1[$.

The [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
presented in [the previous section](active_learning_algo.md)
can be used to *approximate* a particular quantile,
using the family of acquisition criteria
[Quantile][gemseo_mlearning.active_learning.acquisition_criteria.quantile.quantile.Quantile].
These criteria are the same as for the estimation of a level set.
For this reason,
we refer you [to this page](level_set.md) listing these acquisition criteria.

In this page,
we simply indicate that
the quantile level and the uncertain space are set
using the arguments `level` and `uncertain_space` respectively,
and the user code looks like below.

## U-function

```python
active_learning = ActiveLearningAlgo(
    "Quantile",
    input_space,
    initial_regressor,
    level=0.1,
    uncertain_space=probability_space
)
```

## Expected feasibility

```python
active_learning = ActiveLearningAlgo(
    "Quantile",
    input_space,
    initial_regressor,
    criterion_name="EF",
    level=0.1,
    uncertain_space=probability_space
)
```

## Expected improvement

```python
active_learning = ActiveLearningAlgo(
    "Quantile",
    input_space,
    initial_regressor,
    criterion_name="EI",
    level=0.1,
    uncertain_space=probability_space
)
```
