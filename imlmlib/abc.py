import pymc
from imlmlib.mem_utils import BlockBasedSchedule, experiment
from imlmlib.exponential_forgetting import GaussianEFPopulation
import functools
import matplotlib.pyplot as plt
import numpy


def ef_simulator(schedule, population_kwargs, rng, log10_alpha, beta, size = None):

    population_kwargs.update({"mu_a": 10**(log10_alpha), "mu_b": beta})

    population_model = GaussianEFPopulation(**population_kwargs)
    data = experiment(population_model, schedule, replications=1)
    nblock = len(schedule.interblock_time) + 1

    data = (
        data[0, 0, ...]
        .transpose(1, 0)
        .reshape(
            population_model.pop_size,
            nblock,
            schedule.nitems * schedule.repet_trials,
        )
    )
    return data.mean(axis=(0, 2)).squeeze()


def ef_infer_abc(schedule, population_kwargs, observed_data, simulator_kwargs = None):
    sim_kwargs = {'epsilon':.01, 'observed': observed_data, 'distance': 'laplace', 'sum_stat' : 'sort', 'ndims_params' : [0,0]}
    if simulator_kwargs is not None:
        sim_kwargs.update(simulator_kwargs)

    with pymc.Model() as _model:
        a = pymc.Uniform('log10alpha', -6,-.5)
        b = pymc.Uniform('b', .01, .99)
        sim = functools.partial(ef_simulator, schedule, population_kwargs)
        s = pymc.Simulator("s", sim, params=(a, b),**sim_kwargs)
        idata = pymc.sample_smc(parallel = True)
        idata.extend(pymc.sample_posterior_predictive(idata))

    return idata

def plot_ihd_contours(idata, ax = None):
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
    log10_alpha_posterior = numpy.concatenate(idata.posterior['log10alpha'].values)
    beta_posterior = numpy.concatenate(idata.posterior['b'].values)
    az.plot_kde(log10_alpha_posterior, values2=beta_posterior, ax = ax, hdi_probs = [.68, .95], contourf_kwargs = {'colors': ['#B0E0E6', '#87CEEB']}, legend = False)
    return ax