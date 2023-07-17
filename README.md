# imlmlib
This imlmlib, an inference, simulation and utility library for interaction modalities that leverage memory. 

# building/using the library

1. You can download the source and use the library with poetry
2. You can download the source and install it with pip
3. You can install the module directly from [PyPI](https://pypi.org/project/imlmlib/)

```bash
pip install imlmlib
```



# Infering memory model parameters


## Maximum Likelihood Estimation

In general, MLE for a recall sequences will look like this:

```python
from imlmlib.mle_utils import (
    identify_alpha_beta_from_recall_sequence,
)

# example sequences used for inference
recall = [0,1,0,...,0]
deltas = [numpy.infty, 100, 100, ..., 100]
k = [-1, 0, 2, ..., N]

## warning: k_vector is number of repetitions and not number of presentations of the item. By convention, a first presentation should have k = -1 (for which delta = -numpy.infty and recall = 0 always)
inference_results,
                J,
                CI_alpha,
                CI_beta,
            ) = identify_alpha_beta_from_recall_sequence(
                recall,
                deltas,
                k_vector=k_vector,
                optim_kwargs=optim_kwargs,
                verbose=verbose,
                guess=guess,
                CI=True,
            )

## inference_results is the array of estimated parameters
## J is the observed information matrix
## CI_alpha, CI_beta are the 95\% asymptotic confidence intervals computed by inverting J


## If CI = False, then only inference_results is returned
## if k_vector is None, range(N) is constructed. This means you should remove the first item presentation in recall and delta.

```

## worked out example

```python

# Get data from a dataset or any data generating process
# By convention, The first time an item is seen, recall = False (0), delta = numpy.infty and k = -1 The second time an item is seen, the first repetition is k = 0. 
deltas, recalls, repetitions = ()
r = recalls
dt = deltas 
k = repetitions 


# maximum likelihood estimate
# if k_vector is None (default), k = range(len(r))
results = identify_alpha_beta_from_recall_sequence(
            [int(_r) for _r in r], dt, guess=(1e-3, 0.5), k_vector=k
        )
# compute Fischer's observed information matrix
J = observed_information_matrix([int(_r) for _r in r], dt, *results.x, k_vector=k)

# use J to compute the confidence ellipse. This can be done in log scale for alpha (alpha_scale = 'log'). With CI95= True, will also return the (marginal) confidence intervals.
ax, ci_a, ci_b = confidence_ellipse(results.x, numpy.linalg.inv(J), alpha_scale='log', plot = True, CI95=True)
print(ci_a, ci_b)

# print diagnostics to asses model fit. Needs enough data, and sufficiently spread out.
fig, (fg, ax, estim) = diagnostics(*results.x, k, dt, r)
print(estim)
plt.show()
exit()
```


## Approximate Bayesian Computation

```python

from imlmlib.mem_utils import BlockBasedSchedule
from imlmlib.abc import ef_infer_abc, ef_simulator, plot_ihd_contours
from imlmlib.mem_utils import BlockBasedSchedule
import arviz as az
import matplotlib.pyplot as plt
# recreate the experiment conditions: number of participants and number of items.
population_kwargs = {
        "population_size": 24,
        "n_items": 15,
        "seed": None,
        "mu_a": None,
        "sigma_a": 1e-2 / 6,
        "mu_b": None,
        "sigma_b": 0.5 / 6,
    }

# create the schedule used in the experiment. For example, here we have 15 items, with an intertrial time of 5 seconds and inter block durations of [2000, 2000, 86400, 2000]. Repet_trials = 2 means the schedule is repeated twice per block (i.e. the 15 items are seen twice within a block in the same order)
schedule = BlockBasedSchedule(15, 5, [2000, 2000, 86400, 2000], repet_trials=2)
observed_data = [] # --> write down your observed recalls here
simulator_kwargs ={"epsilon": .1} # set rejection threshold for abc. In general, here you would fit all kwargs for the pymc.Simulator object.
# perform the inference
idata = ef_infer_abc(schedule, population_kwargs, observed_data, simulator_kwargs)

# various useful outputs
print(az.summary(idata, kind="stats"))
az.plot_forest(idata, var_names=["log10alpha", 'b'], combined=True, hdi_prob=0.95)

ax = plot_ihd_contours(idata)
plt.show()

```

## Memory Models

### Exponential forgetting
Initialise the memory model:
```python

model = ExponentialForgetting(a = 0.8, b = 0.8)

```
The model history can be cleared with

```python

model.reset()

```
New items can be presented to the model. For example, one can present an item at a specific time using
```python

model.update(item, time)

```

The model can also be queried for an item at a given time, in which case the model returns a boolean indicating (un)successful recall of the queried item, as well as the model's probability of correctly recalling the item.

```python

recall, recall_probability = model.query(item, time)

```

You can also create a population of user models. For now, there is only a Gaussian population.
For example, with 

```python

population = GaussianEFPopulation(N, mu_a = 0.8, sigma_a = 0.01, mu_b = 0.8, sigma_b = 0.01, seed = 1234)

```
you create an object that is iterable N times (ie as if there where N different participants), where each item (each participant) is an ExponentialForgetting memory model where its a and b parameters are drawn from two Gaussians with respectively parameters mu\_a and sigma\_a, and mu\_b and sigma\_b.

### ACT-R 2005
To arrive

## Simulation
There are various utilities that are useful for simulation

### Schedule
The Schedule class pairs timestamps with items to query.

### trial
trial applies a schedule to a memory model
```python

queries, updated_memory_model = trial(memory_model, schedule)

```
### experiment
experiment applies trials to a population model according to a given schedule, with R replications 

```python

data = experiment(population_model, schedule, replications=R)

```

The output is an [R, 2, len(schedule), N] array. The 2nd dimension corresponds to a query (recall, recall_probability)


