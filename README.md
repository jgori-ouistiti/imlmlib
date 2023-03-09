# imlmlib
This imlmlib, an inference, simulation and utility library for interaction modalities that leverage memory. 

# building/using the library

1. You can download the source and use the library with poetry
2. You can download the source and install it with pip (use -e if you want to edit)
3. You can download the module directly from [PyPI](https://pypi.org/project/imlmlib/)

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

## Maximum Likelihood estimation
There are utilities to infer a and b parameters from empirical data. If recall data is available, then you can do 
```python
# If schedule
times = schedule.times
# else times could be simply a list of timestamps

# if simulated data using experiment for a single participant and a single trial
recall_single_trial = data[0, 0, :, 0]
# else, could just pass a list of recalls

infer_results = estim_mle_one_trial(
    times,
    recall_single_trial,
    ef_get_per_participant_likelihood_log_a,
    {"method": "SLSQP", "bounds": [(-5, -1), (0, 0.99)]}, # bounds on possible a and b values
    (-3, 0.8), # initial guess
)

```

