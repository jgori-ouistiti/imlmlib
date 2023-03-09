import imlmlib.exponential_forgetting as EF


def test_EF():
    recalls = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ks = [i for i in range(len(recalls))]
    deltats = [100, 100, 100, 1700, 200, 200, 1600, 100, 100, 3800]
    a = -1.1
    b = 0.6
    ll = 0
    for recall, k, deltat in zip(recalls, ks, deltats):
        res1 = EF.ef_log_likelihood_sample_log_a(1, k, deltat, a, b)
        res2 = EF.ef_log_likelihood_sample_log_a(0, k, deltat, a, b)
        ll += EF.ef_log_likelihood_sample_log_a(recall, k, deltat, a, b)


if __name__ == "__main__":
    test_EF()
