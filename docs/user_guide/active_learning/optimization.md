<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Active learning for optimization

[Active learning](what_active_learning_is.md) techniques have become popular in global optimization,
with the famous Bayesian optimization algorithms,
including the efficient global optimization (EGO) one[@jones1998efficient].
These approaches can be relevant
when the functions of the optimization problem:

- are costly,
- are not accompanied by their gradient functions,
- depend on an important number of optimization variables,
  making the estimation of their gradient
  by finite differences impossible.

In this case,
multi-start gradient-based optimization cannot be used,
and neither can classical global optimization algorithms (*e.g.* evolutionary algorithms).

The [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
presented in [the previous section](active_learning_algo.md)
can be used to minimize (resp. maximize) a cost function (resp. performance function)
using the family of acquisition criteria
[Minimum][gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum.Minimum]
(resp.
[Maximum][gemseo_mlearning.active_learning.acquisition_criteria.maximum.maximum.Maximum]
).
The quantity of interest is the minimum (resp. maximum) of the function.

Given a random process $Y$ modelling the uncertainty of the surrogate model $\hat{f}$,
this page lists several acquisition criteria,
first for the minimum,
then for the maximum.
Most often,
$Y$ is a Gaussian process (GP) conditioned by the training dataset
and $\hat{f}$ is the associated GP regressor,
which corresponds to its expectation.
Given an input point $x$,
$\mu(x)\equiv\mathbb{E}[Y(x)]$ and $\sigma(x)\equiv\mathbb{S}[Y(x)]$
denote the expectation and the standard deviation of $Y(x)$.

For acquisition criteria supporting parallel acquisition,
the number of points to be acquired at a time
and the sample size to estimate the statistics defined below
can be set with the arguments `batch_size` (default: `1`) and `mc_size` (default: `10000`) respectively.

## Minimum

### Mean

The simplest criterion to _minimize_ is the expectation:

$$E[x] = \mu(x),$$

also called Kriging believer in the case of a Gaussian process regressor.

#### API

This criterion can be accessed via the name `"Mean"`:

```python
active_learning = ActiveLearningAlgo(
    "Minimum",
    input_space,
    initial_regressor,
    criterion_name="Mean"
)
```

### Lower confidence bound

A more advanced criterion to _minimize_ is the lower confidence bound

$$M[x;\kappa] = \mu(x) - \kappa \times \sigma(x)$$

where $\kappa > 0$ (default: 2).

#### API

This criterion can be accessed via the name `"LCB"`:

```python
active_learning = ActiveLearningAlgo(
    "Minimum",
    input_space,
    initial_regressor,
    criterion_name="LCB"
)
```

and the $\kappa$ constant can be set using the argument `kappa`.

### Expected improvement

The most popular criterion to _maximize_ is the expected improvement[@jones1998efficient]

$$EI[x] = \mathbb{E}[\max(\min(y_1,\dots,y_n)-Y(x),0)].$$

where $y_1,\dots,y_n$ are the learning output values.

#### API

This criterion, named `"EI"`, is the default one;
so there's no need to provide its name:

```python
active_learning = ActiveLearningAlgo(
    "Minimum",
    input_space,
    initial_regressor
)
```

#### Gaussian case

In the case of a Gaussian process regressor,
it has an analytic expression:

$$EI[x] = (\min(y_1,\dots,y_n)-\mu(x))\Phi(t) + \sigma(x)\phi(t)$$

where $\Phi$ and $\phi$ are respectively
the cumulative and probability density functions
of the standard normal distribution
and $t=\frac{y_{\text{min}}-\mu(x)}{\sigma(x)}$.

#### Parallel acquisition

For the acquisition of $q>1$ points at a time,
the acquisition criterion changes to

$$
EI[x_1,\dots,x_q] =
\mathbb{E}\left[\max_{1\leq i\leq q}\left(\max(\min(y_1,\dots,y_n)-Y(x_i),0)\right)\right]
$$

where the expectation is taken with respect to the distribution of
the random vector $(Y(x_1),\dots,Y(x_q))$.
There is no analytic expression
and the acquisition criterion is thus instead evaluated with crude Monte Carlo.

## Maximum

### Mean

The simplest criterion to _maximize_ is the expectation

$$E[x] = \mu(x),$$

also called Kriging believer in the case of a Gaussian process regressor.

#### API

This criterion can be accessed via the name `"Mean"`:

```python
active_learning = ActiveLearningAlgo(
  "Maximum",
  input_space,
  initial_regressor,
  criterion_name="Mean"
)
```

### Upper confidence bound

A more advanced criterion to _maximize_ is the upper confidence bound

$$M[x;\kappa] = \mu(x) + \kappa \times \sigma(x)$$

where $\kappa > 0$ (default: 2).

#### API

This criterion can be accessed via the name `"UCB"`:

```python
active_learning = ActiveLearningAlgo(
    "Minimum",
    input_space,
    initial_regressor,
    criterion_name="LCB"
)
```

and the $\kappa$ constant can be set using the argument `kappa`.

### Expected improvement

The most popular criterion to _maximize_ is the expected improvement[@jones1998efficient]

$$EI[x] = \mathbb{E}[\max(Y(x)-\max(y_1,\dots,y_n)},0)]$$

where $y_1,\dots,y_n$ are the learning output values.

#### API

This criterion, named `"EI"`, is the default one;
so there's no need to provide its name:

```python
active_learning = ActiveLearningAlgo(
    "Maximum",
    input_space,
    initial_regressor
)
```

#### Gaussian case

In the case of a GP,
the criterion has an analytic expression:

$$EI[x] = (\mu(x) - \max(y_1,\dots,y_n)})\Phi(t) + \sigma(x)\phi(t)$$

where $\Phi$ and $\phi$ are respectively
the cumulative and probability density functions
of the standard normal distribution
and $t=\frac{\mu(x) - \max(y_1,\dots,y_n)}}{\sigma(x)}$.

#### Parallel acquisition

For the acquisition of $q>1$ points at a time,
the acquisition criterion changes to

$$
EI[x_1,\dots,x_q] =
\mathbb{E}\left[\max_{1\leq i\leq q}\left(\max(Y(x_i)-\max(y_1,\dots,y_n)},0)\right)\right]
$$

where the expectation is taken with respect to the distribution of
the random vector $(Y(x_1),\dots,Y(x_q))$.
There is no analytic expression
and the acquisition criterion is thus instead evaluated with crude Monte Carlo.
