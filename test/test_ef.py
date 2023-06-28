import imlmlib.exponential_forgetting as EF
import imlmlib.mem_utils as MU
import imlmlib.mle_utils as MLU
import numpy


def test_EF_sample():
    ef = EF.ExponentialForgetting(1, a=1e-2, b=0.4)
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
        10000,
    ]
    items = [0 for i in times]
    ef.reset()
    assert ef.counters[0, 0] == 0
    assert ef.counters[0, 1] == -numpy.inf

    for n, (item, time) in enumerate(zip(items, times)):
        assert ef.counters[0, 0] == n
        ef.query_item(item, time)
        ef.update(item, time)
        assert ef.counters[0, 0] == n + 1
        assert ef.counters[0, 1] == time


def test_EF_notransform():
    recalls = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ks = [i for i in range(len(recalls))]
    deltats = [100, 100, 100, 1700, 200, 200, 1600, 100, 100, 3800]
    a = 1e-2
    b = 0.6
    ll = 0
    for recall, k, deltat in zip(recalls, ks, deltats):
        transform = lambda a, b: (a, b)
        res1 = EF.ef_log_likelihood_sample(1, k, deltat, a, b, transform)
        res2 = EF.ef_log_likelihood_sample(0, k, deltat, a, b, transform)
        ll += EF.ef_log_likelihood_sample(recall, k, deltat, a, b, transform)


def test_pop_model():
    population_model = EF.GaussianEFPopulation(
        1, 1, seed=None, mu_a=1e-2, mu_b=0.4, sigma_a=1e-2 / 10000, sigma_b=0.5 / 10000
    )


def test_rng_pop():
    population_model_one = EF.GaussianEFPopulation(
        1, 1, seed=123, mu_a=1e-2, mu_b=0.4, sigma_a=None, sigma_b=None
    )
    population_model_two = EF.GaussianEFPopulation(
        1, 1, seed=123, mu_a=1e-2, mu_b=0.4, sigma_a=None, sigma_b=None
    )

    times = numpy.linspace(0, 1000, 10)
    items = [0 for i in times]  # only one item
    REPL = 100
    schedule_one = MU.Schedule(items, times)  # make schedule
    data_one = MU.experiment(population_model_one, schedule_one, replications=REPL)
    data_two = MU.experiment(population_model_two, schedule_one, replications=REPL)
    assert (data_one == data_two).all()


def test_init_gefp():
    gefp = EF.GaussianEFPopulation(
        3, 1, mu_a=1e-3, mu_b=0.4, sigma_a=1e-6, sigma_b=1e-6
    )
    assert gefp.mu_a == 1e-3
    assert gefp.mu_b == 0.4
    assert gefp.sigma_a == 1e-6
    assert gefp.sigma_b == 1e-6


def test_gen_gefp():
    N = 3
    gefp = EF.GaussianEFPopulation(
        N, 1, mu_a=1e-3, mu_b=0.4, sigma_a=1e-6, sigma_b=1e-6
    )
    for n, ef in enumerate(gefp):
        assert isinstance(ef, EF.ExponentialForgetting)
        assert numpy.abs((ef.a - 1e-3)) < 1e-3
        assert numpy.abs((ef.b - 0.4)) < 1e-3
    assert n == N - 1


def test_efll_with_first():
    alpha = 1e-2
    beta = 0.4
    ef_model = EF.ExponentialForgetting(1, a=alpha, b=beta)
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
        10000,
    ]
    sched = MU.Schedule(
        [0 for i in range(12)],
        times,
    )
    queries, memory_model = MU.trial(ef_model, sched)
    recall = [z[0] for z in queries]
    deltas = numpy.zeros((len(times),))
    deltas[0] = numpy.inf
    deltas[1:] = numpy.diff(times)
    k_vector = range(-1, len(recall) - 1)

    for n, (k, recall) in enumerate(zip(k_vector, recall)):
        dll = EF.ef_log_likelihood_sample(
            recall, k, deltas[n], alpha, beta, transform=lambda a, b: (a, b)
        )
        if recall:
            assert numpy.abs(numpy.exp(dll) - queries[n][1]) < 1e-6
        else:
            assert numpy.abs(1 - numpy.exp(dll) - queries[n][1]) < 1e-6


def test_efll_without_first():
    alpha = 1e-2
    beta = 0.4
    ef_model = EF.ExponentialForgetting(1, a=alpha, b=beta)
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
        10000,
    ]
    sched = MU.Schedule(
        [0 for i in range(12)],
        times,
    )
    queries, memory_model = MU.trial(ef_model, sched)
    recall = [z[0] for z in queries]
    recall = recall[1:]
    deltas = numpy.diff(times)

    for nsched, recall in enumerate(recall):
        dll = EF.ef_log_likelihood_sample(
            recall, nsched, deltas[nsched], alpha, beta, transform=lambda a, b: (a, b)
        )
        if recall:
            assert numpy.abs(numpy.exp(dll) - queries[nsched + 1][1]) < 1e-6
        else:
            assert numpy.abs(1 - numpy.exp(dll) - queries[nsched + 1][1]) < 1e-6


def test_duplicate():
    alpha = 1e-2
    beta = 0.4
    ef_model = EF.ExponentialForgetting(1, a=alpha, b=beta)
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
        10000,
    ]
    sched = MU.Schedule(
        [0 for i in range(12)],
        times,
    )
    queries, memory_model = MU.trial(ef_model, sched)
    recall = [z[0] for z in queries]

    deltas = numpy.zeros((len(times),))
    deltas[0] = numpy.inf
    deltas[1:] = numpy.diff(times)
    k_vector = list(range(-1, len(recall) - 1))

    data = recall + recall
    k_vector = k_vector + k_vector
    deltas = deltas.tolist() + deltas.tolist()

    for n, (k, recall) in enumerate(zip(k_vector, data)):
        dll = EF.ef_log_likelihood_sample(
            recall, k, deltas[n], alpha, beta, transform=lambda a, b: (a, b)
        )
        if recall:
            assert numpy.abs(numpy.exp(dll) - queries[n % 12][1]) < 1e-6
        else:
            assert numpy.abs(1 - numpy.exp(dll) - queries[n % 12][1]) < 1e-6

    optim_kwargs = {"method": "L-BFGS-B", "bounds": [(1e-5, 0.1), (0, 0.99)]}

    verbose = False
    guess = (1e-3, 0.5)
    inference_results = MLU.identify_alpha_beta_from_recall_sequence(
        data,
        deltas,
        k_vector=k_vector,
        optim_kwargs=optim_kwargs,
        verbose=verbose,
        guess=guess,
    )
    print(inference_results)


if __name__ == "__main__":
    test_EF_sample()
    test_EF_notransform()
    test_pop_model()
    test_init_gefp()
    test_gen_gefp()
    test_efll_with_first()
    test_efll_without_first()
    test_duplicate()
    test_rng_pop()
