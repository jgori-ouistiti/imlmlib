import numpy
import scipy.optimize as opti

from imlmlib.exponential_forgetting import (
    ef_get_per_participant_likelihood_log_a,
    GaussianEFPopulation,
)
from imlmlib.mem_utils import Schedule, experiment
from imlmlib.mle_utils import estim_mle_one_trial

REPLICATIONS = 1
times = [
    0,
    100,
    200,
    300,
    2000,
    2200,
    2400,
    4000,
    4100,
    4200,
    8000,
]
items = [0 for i in times]  # only one item
schedule_one = Schedule(items, times)  # make schedule
population_model = GaussianEFPopulation(
    3, 1, seed=None, mu_a=1e-2, mu_b=0.5, sigma_a=1e-2 / 100, sigma_b=0.5 / 100
)
data = experiment(population_model, schedule_one, replications=REPLICATIONS)


def test_one_trial():
    times = schedule_one.times
    recall_single_trial = data[0, 0, :, 0]
    infer_results = estim_mle_one_trial(
        times,
        recall_single_trial,
        ef_get_per_participant_likelihood_log_a,
        {"method": "SLSQP", "bounds": [(-5, -1), (0, 0.99)]},
        (-3, 0.8),
    )
    return


if __name__ == "__main__":
    test_one_trial()
