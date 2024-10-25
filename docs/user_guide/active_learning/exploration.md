<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Active learning for exploration

[Active learning](what_active_learning_is.md) techniques are very intuitive for refining a surrogate model.
Instead of spending the entire computational budget to create a training dataset in one-go,
the idea is to start from a rough surrogate model and acquire new training points sequentially to increase its accuracy.

The [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
presented in [the previous section](active_learning_algo.md)
can be used to *explore* the input space in search of unlearned points,
using the family of acquisition criteria
[Exploration][gemseo_mlearning.active_learning.acquisition_criteria.exploration.exploration.Exploration].

For now,
these acquisition criteria do not have quantities of interest
and do not support parallel acquisition.

Given a random process $Y$ modelling the uncertainty of the surrogate model $\hat{f}$,
this page lists several acquisition criteria.
Most often,
$Y$ is a Gaussian process (GP) conditioned by the training dataset
and $\hat{f}$ is the associated GP regressor,
which corresponds to its expectation.
Whatever surrogate model is chosen,
given an input point $x$,
$\mu(x)\equiv\mathbb{E}[Y(x)]$ and $\sigma(x)\equiv\mathbb{S}[Y(x)]$
denote the expectation and the standard deviation of $Y(x)$.

## Distance

The most naive approach is to learn the point furthest away from those in the training dataset.
So,
this criterion to _maximize_ is
the minimum distance between a new point
and the point of the training dataset,
scaled by the minimum distance between two distinct training points:

$$D[x]=\frac{\min_{1\leq i \leq n} \|x-x_i\|_2}{\min_{1\leq i,j \leq n, i\neq j} \|x_i-x_j\|_2}$$

where $\|.\|_2$ is the Euclidean norm.

#### API

This criterion can be accessed via the name `"Distance"`:

```python
active_learning = ActiveLearningAlgo(
    "Exploration",
    input_space,
    initial_regressor,
    criterion_name="Distance"
)
```

## Variance

The previous criterion considers neither the output values nor the quality of the surrogate model.
However,
taking account of model uncertainty can be relevant
and this is the reason why the variance of the prediction is the default criterion:

$$V[x]=\sigma^2(x).$$

#### API

This criterion, named `"Variance"`,
is the default one;
so there's no need to provide its name:

```python
active_learning = ActiveLearningAlgo(
    "Exploration",
    input_space,
    initial_regressor,
)
```

## Standard deviation

This criterion is simply the square root of the previous one:

$$S[x]=\sigma(x).$$

#### API

This criterion can be accessed via the name `"StandardDeviation"`:

```python
active_learning = ActiveLearningAlgo(
    "Exploration",
    input_space,
    initial_regressor,
    criterion_name="StandardDeviation"
)
```
