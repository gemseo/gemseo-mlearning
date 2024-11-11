<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Active learning for level set estimation

[Active learning](what_active_learning_is.md) techniques are very useful
to estimate a level set, _i.e._ the input values for which the model output is equal to a specific value $y$.

The [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
presented in [the previous section](active_learning_algo.md)
can be used to *approximate* a particular level set,
using the family of acquisition criteria
[LevelSet][gemseo_mlearning.active_learning.acquisition_criteria.level_set.level_set.LevelSet].
As new points are acquired,
the level set of the surrogate model will be increasingly similar to that of the substituted model.

For now, these acquisition criteria do not have quantities of interest.

Given a random process $Y$ modelling the uncertainty of the surrogate model $\hat{f}$,
this page lists several acquisition criteria.
Most often,
$Y$ is a Gaussian process (GP) conditioned by the training dataset
and $\hat{f}$ is the associated GP regressor,
which corresponds to its expectation.
Given an input point $x$,
$\mu(x)\equiv\mathbb{E}[Y(x)]$ and $\sigma(x)\equiv\mathbb{S}[Y(x)]$
denote the expectation and the standard deviation of $Y(x)$.

The value characterizing the level set can be passed using the argument `output_value`,
 _e.g._ `output_value=12`.

All these acquisition criteria support parallel acquisition.
The number of points to be acquired at a time
and the sample size to estimate the statistics defined below
can be set with the arguments `batch_size` (default: `1`)
and `mc_size` (default: `10000`) respectively.

## U-function

The simplest criterion to approximate a level set $y$ is the U-function[@roy2014estimating]:

$$U[x] = \mathbb{E}\left[\left(\frac{y-Y(x)}{\sigma(x)}\right)^2\right]$$

which can be simplified to

$$U[x] = \left(\frac{y-\mu(x)}{\sigma(x)}\right)^2.$$

### API

This criterion, named `"U"`, is the default one;
so there's no need to provide its name:

```python
active_learning = ActiveLearningAlgo(
    "LevelSet",
    input_space,
    initial_regressor,
    output_value=12.
)
```

### Parallel acquisition

For the acquisition of $q>1$ points at a time,
the acquisition criterion changes to

$$
U[x_1,\dots,x_q] =
\mathbb{E}\left[\min_{1\leq i \leq q}\left(\left(\frac{y-Y(x_i)}{\sigma(x_i)}\right)^2\right)\right]
$$

where the expectation is taken with respect to the distribution of
the random vector $(Y(x_1),\dots,Y(x_q))$.
There is no analytic expression
and the acquisition criterion is thus instead evaluated with crude Monte Carlo.

## Expected feasibility

Another criteria to approximate a level set $y$ is the expected feasibility[@bect2012sequential]:

$$EF[x] =  \mathbb{E}\left[\max(\kappa\sigma(x) - |y - Y(x)|,0)\right].$$

### API

This criterion can be accessed via the name `"EF"`:

```python
active_learning = ActiveLearningAlgo(
    "LevelSet",
    input_space,
    initial_regressor,
    criterion_name="EF",
    output_value=12.
)
```

### Gaussian case

In the case of a GP,
the criterion has an analytic expression:

$$
EF[x] =
\sigma(x)
\left(
\kappa
(\Phi(t^+)-\Phi(t^-))
-t(2\Phi(t)-\Phi(t^+)-\Phi(t^-))
-(2\phi(t)-\phi(t^+)-\phi(t^-))
\right)
$$

where $\Phi$ and $\phi$ are respectively
the cumulative and probability density functions
of the standard normal distribution,
$t=\frac{y - \mu(x)}{\sigma(x)}$,
$t^+=t+\kappa$,
$t^-=t-\kappa$
and $\kappa>0$.

### Parallel acquisition

For the acquisition of $q>1$ points at a time,
the acquisition criterion changes to

$$
EF[x_1,\dots,x_q] =
\mathbb{E}\left[\max_{1\leq i \leq q}\left(\max(\kappa\sigma(x_i) - |y - Y(x_i)|,0)\right)\right]
$$

where the expectation is taken with respect to the distribution of
the random vector $(Y(x_1),\dots,Y(x_q))$.
There is no analytic expression
and the acquisition criterion is thus instead evaluated with crude Monte Carlo.

## Expected improvement

Another criteria to approximate a level set $y$ is the expected improvement[@bect2012sequential]:

$$EI[x] = \mathbb{E}\left[\max((\kappa\sigma(x))^2 - (y - Y(x))^2,0)\right].$$

### API

This criterion can be accessed via the name `"EI"`:

```python
active_learning = ActiveLearningAlgo(
    "LevelSet",
    input_space,
    initial_regressor,
    criterion_name="EI",
    output_value=12.
)
```

### Gaussian case

In the case of a GP,
the criterion has an analytic expression:

$$
EI[x] = \sigma(x)
\left(
(\kappa^2-1-t^2)(\Phi(t^+)-\Phi(t^-))
-2t(\phi(t^+)-\phi(t^-))
+t^+\phi(t^+)
-t^-\phi(t^-)
\right)
$$

where $Y$ is the random process
modelling the uncertainty of the surrogate model $\hat{f}$,
$t=\frac{y - \mu(x)}{\sigma(x)}$,
$t^+=t+\kappa$,
$t^-=t-\kappa$
and $\kappa>0$.

### Parallel acquisition

For the acquisition of $q>1$ points at a time,
the acquisition criterion changes to

$$
EI[x_1,\dots,x_q] =
\mathbb{E}\left[\max_{1\leq i\leq q}\left(\max((\kappa\sigma(x_i))^2 - (y - Y(x_i))^2,0)\right)\right]
$$

where the expectation is taken with respect to the distribution of
the random vector $(Y(x_1),\dots,Y(x_q))$.
There is no analytic expression
and the acquisition criterion is thus instead evaluated with crude Monte Carlo.
