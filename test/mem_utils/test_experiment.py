from imlmlib.mem_utils import experiment, trial
from imlmlib.exponential_forgetting import GaussianEFPopulation
from imlmlib.mem_utils import Schedule

import numpy


def test_experiment():
    population_model = GaussianEFPopulation(
        1, 1, seed=None, mu_a=1e-2, mu_b=0.5, sigma_a=1e-2 / 100, sigma_b=0.5 / 100
    )
    sched = Schedule(
        [0 for i in range(10)], [125, 250, 375, 500, 1000, 1500, 2000, 2500, 3000, 4000]
    )
    data = experiment(population_model, sched, 1000)


if __name__ == "__main__":
    test_experiment()
