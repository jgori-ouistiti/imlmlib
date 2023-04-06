import imlmlib.exponential_forgetting as EF


def test_EF():
    recalls = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ks = [i for i in range(len(recalls))]
    deltats = [100, 100, 100, 1700, 200, 200, 1600, 100, 100, 3800]
    a = -1.1
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


if __name__ == "__main__":
    test_EF()
    test_pop_model()
