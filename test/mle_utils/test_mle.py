from imlmlib.exponential_forgetting import (
    GaussianEFPopulation,
)
from imlmlib.mem_utils import Schedule, experiment
from imlmlib.mle_utils import sample_mle

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


def test_mle():
    REPL = 1
    Nseq = 1
    results, _ = sample_mle(REPL, Nseq, population_model, schedule_one)


if __name__ == "__main__":
    test_mle()
