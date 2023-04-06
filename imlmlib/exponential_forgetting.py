import warnings

warnings.filterwarnings("error")
import numpy


from imlmlib.mem_utils import MemoryModel

my_diff = lambda schedule: numpy.diff([s[1] for s in schedule])


## ============ for observed information matrix ============= ##
## p1, q1, p0, q0
def ef_p1_sample(alpha, beta, k, deltat):
    return numpy.exp(ef_q1_sample(alpha, beta, k, deltat))


def ef_p0_sample(alpha, beta, k, deltat):
    return 1 - ef_p1_sample(alpha, beta, k, deltat)


def ef_q1_sample(alpha, beta, k, deltat):
    return -alpha * (1 - beta) ** k * deltat


def ef_q0_sample(alpha, beta, k, deltat):
    return numpy.log(1 - numpy.exp(ef_q1_sample(alpha, beta, k, deltat)))


## first order derivatives
def ef_dq1_dalpha_sample(alpha, beta, k, deltat):
    return -((1 - beta) ** k) * deltat


def ef_dq1_dbeta_sample(alpha, beta, k, deltat):
    return alpha * k * (1 - beta) ** (k - 1) * deltat


def ef_dq0_alpha_sample(alpha, beta, k, deltat):
    return (
        (1 - beta) ** k
        * deltat
        * ef_p1_sample(alpha, beta, k, deltat)
        / ef_p0_sample(alpha, beta, k, deltat)
    )


def ef_dq0_beta_sample(alpha, beta, k, deltat):
    return (
        -k
        * alpha
        * (1 - beta) ** (k - 1)
        * deltat
        * ef_p1_sample(alpha, beta, k, deltat)
        / ef_p0_sample(alpha, beta, k, deltat)
    )


# test with sympy

# q1
def __sym_ef_dq1_dalpha_sample(a, b, k, d):
    return -d * (1 - b) ** k


def __sym_ef_dq1_dbeta_sample(a, b, k, d):
    return a * d * k * (1 - b) ** k / (1 - b)


def __sym_ef_ddq1_dalpha_dalpha_sample(a, b, k, d):
    return 0


def __sym_ef_ddq1_dalpha_dbeta_sample(a, b, k, d):
    return d * k * (1 - b) ** k / (1 - b)


def __sym_ef_ddq1_dbeta_dbeta_sample(a, b, k, d):
    return (
        -a * d * k**2 * (1 - b) ** k / (1 - b) ** 2
        + a * d * k * (1 - b) ** k / (1 - b) ** 2
    )


# q0
def __sym_ef_dq0_dalpha_sample(a, b, k, d):
    return (
        d
        * (1 - b) ** k
        * numpy.exp(-a * d * (1 - b) ** k)
        / (1 - numpy.exp(-a * d * (1 - b) ** k))
    )


def __sym_ef_dq0_dbeta_sample(a, b, k, d):
    return (
        -a
        * d
        * k
        * (1 - b) ** k
        * numpy.exp(-a * d * (1 - b) ** k)
        / ((1 - b) * (1 - numpy.exp(-a * d * (1 - b) ** k)))
    )


def __sym_ef_ddq0_dalpha_dalpha_sample(a, b, k, d):
    return (
        -(d**2)
        * (1 - b) ** (2 * k)
        * numpy.exp(-a * d * (1 - b) ** k)
        / (1 - numpy.exp(-a * d * (1 - b) ** k))
        - d**2
        * (1 - b) ** (2 * k)
        * numpy.exp(-2 * a * d * (1 - b) ** k)
        / (1 - numpy.exp(-a * d * (1 - b) ** k)) ** 2
    )


def __sym_ef_ddq0_dalpha_dbeta_sample(a, b, k, d):
    return (
        a
        * d**2
        * k
        * (1 - b) ** (2 * k)
        * numpy.exp(-a * d * (1 - b) ** k)
        / ((1 - b) * (1 - numpy.exp(-a * d * (1 - b) ** k)))
        + a
        * d**2
        * k
        * (1 - b) ** (2 * k)
        * numpy.exp(-2 * a * d * (1 - b) ** k)
        / ((1 - b) * (1 - numpy.exp(-a * d * (1 - b) ** k)) ** 2)
        - d
        * k
        * (1 - b) ** k
        * numpy.exp(-a * d * (1 - b) ** k)
        / ((1 - b) * (1 - numpy.exp(-a * d * (1 - b) ** k)))
    )


def __sym_ef_ddq0_dbeta_dbeta_sample(a, b, k, d):
    return (
        -(a**2)
        * d**2
        * k**2
        * (1 - b) ** (2 * k)
        * numpy.exp(-a * d * (1 - b) ** k)
        / ((1 - b) ** 2 * (1 - numpy.exp(-a * d * (1 - b) ** k)))
        - a**2
        * d**2
        * k**2
        * (1 - b) ** (2 * k)
        * numpy.exp(-2 * a * d * (1 - b) ** k)
        / ((1 - b) ** 2 * (1 - numpy.exp(-a * d * (1 - b) ** k)) ** 2)
        + a
        * d
        * k**2
        * (1 - b) ** k
        * numpy.exp(-a * d * (1 - b) ** k)
        / ((1 - b) ** 2 * (1 - numpy.exp(-a * d * (1 - b) ** k)))
        - a
        * d
        * k
        * (1 - b) ** k
        * numpy.exp(-a * d * (1 - b) ** k)
        / ((1 - b) ** 2 * (1 - numpy.exp(-a * d * (1 - b) ** k)))
    )


## Second order derivatives


def ef_ddq1_dalpha_dalpha_sample(alpha, beta, k, deltat):
    return 0


def ef_ddq1_dalpha_dbeta_sample(alpha, beta, k, deltat):
    return k * (1 - beta) ** (k - 1) * deltat


def ef_ddq1_dbeta_dbeta_sample(alpha, beta, k, deltat):
    return -alpha * k * (k - 1) * (1 - beta) ** (k - 2) * deltat


def ef_ddq0_dalpha_dalpha_sample(alpha, beta, k, deltat):
    with numpy.errstate(divide="raise"):
        try:
            return (
                -((1 - beta) ** (2 * k))
                * deltat**2
                * (
                    ef_p1_sample(alpha, beta, k, deltat)
                    / ef_p0_sample(alpha, beta, k, deltat) ** 2
                )
            )
        except FloatingPointError:
            return -1 / alpha**2


def ef_ddq0_dalpha_dbeta_sample(alpha, beta, k, deltat):
    with numpy.errstate(divide="raise"):
        try:
            return (
                -k
                * (1 - beta) ** (k - 1)
                * deltat
                * ef_p1_sample(alpha, beta, k, deltat)
                / ef_p0_sample(alpha, beta, k, deltat)
                + alpha
                * k
                * (1 - beta) ** (2 * k - 1)
                * deltat**2
                * ef_p1_sample(alpha, beta, k, deltat)
                / ef_p0_sample(alpha, beta, k, deltat) ** 2
            )
        except FloatingPointError:
            return 0


def ef_ddq0_dbeta_dbeta_sample(alpha, beta, k, deltat):
    with numpy.errstate(divide="raise"):
        try:
            return (
                alpha
                * k
                * (k - 1)
                * (1 - beta) ** (k - 2)
                * deltat
                * ef_p1_sample(alpha, beta, k, deltat)
                / ef_p0_sample(alpha, beta, k, deltat)
                - alpha**2
                * k**2
                * deltat**2
                * (1 - beta) ** (2 * k - 2)
                * ef_p1_sample(alpha, beta, k, deltat)
                / ef_p0_sample(alpha, beta, k, deltat) ** 2
            )
        except FloatingPointError:
            return -k * (k * beta - 1) / (1 - beta)


def ef_log_likelihood_sample(recall, k, deltat, alpha, beta, transform):

    # rescaling value to linear
    a, b = transform(alpha, beta)

    if recall == 1:  # Warning: passing to array converts recall to float
        return -a * (1 - b) ** k * deltat
    elif recall == 0:
        with numpy.errstate(over="raise"):
            try:
                exp = numpy.exp(-a * (1 - b) ** k * deltat)
                exp = numpy.clip(exp, a_min=0, a_max=1 - 1e-4)
                return numpy.log(1 - exp)
            except FloatingPointError:
                return -1e6
    else:
        raise ValueError(f"Recall is not 0 or 1, but is {recall}")


def ef_get_per_participant_likelihood_transform(
    theta, deltas, recalls, transform, k_vector=None
):
    ll = 0
    alpha, beta = theta

    if k_vector is None:
        for nsched, recall in enumerate(recalls[1:]):
            ll += ef_log_likelihood_sample(
                recall, nsched, deltas[nsched], alpha, beta, transform
            )
    else:
        for n, (k, recall) in enumerate(zip(k_vector, recalls)):
            ll += ef_log_likelihood_sample(recall, k, deltas[n], alpha, beta, transform)
    return ll


class ExponentialForgetting(MemoryModel):
    def __init__(self, nitems, *args, a=0.1, b=0.5, **kwargs):
        super().__init__(nitems, *args, **kwargs)
        self.a, self.b = a, b
        self.reset()

    def reset(self, a=None, b=None):
        super().reset()
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

        self.counters = numpy.zeros((self.nitems, 2))
        self.counters[:, 1] = -numpy.inf

    def update(self, item, time):
        item = int(item)
        self.counters[item, 0] += 1
        self.counters[item, 1] = time

    def _print_info(self):
        print(f"counters: \t {self.counters}")

    def compute_probabilities(self, time=None):

        if time is None:
            time = numpy.max(self.counters[:, 1])
        n = self.counters[:, 0]
        deltat = time - self.counters[:, 1]
        return numpy.exp(-self.a * (1 - self.b) ** (n - 1) * deltat)

    def __repr__(self):
        return f"{self.__class__.__name__}\n a = {self.a}\n b = {self.b}"


class GaussianEFPopulation:
    """A Population with Exponential Forgetting model and its parameters are sampled according to a Gaussian.

    You can iterate over this object until you have used up all its pop_size members.
    """

    def __init__(
        self,
        pop_size,
        n_items,
        *args,
        seed=None,
        mu_a=1e-2,
        sigma_a=1e-2 / 6,
        mu_b=0.5,
        sigma_b=0.5 / 6,
        **kwargs,
    ):
        """__init__

        :param pop_size: size of the population. You can iterate over the population until you reach its size.
        :type pop_size: int
        :param seed: Seed for the RNG used to draw members of the population
        :type seed: numpy seed-like
        :param mu_a: mean of the a parameter, defaults to 1e-2
        :type mu_a: float, optional
        :param sigma_a: standard deviation of the a parameter, defaults to 1e-2/6
        :type sigma_a: float, optional
        :param mu_b: mean of the b parameter, defaults to 0.5
        :type mu_b: float, optional
        :param sigma_b: standard deviation of the b parameter, defaults to 0.5/6
        :type sigma_b: float, optional
        """
        self.pop_size = pop_size
        self.seed = numpy.random.SeedSequence(seed)
        self.n_items = n_items
        self.args = args
        self.kwargs = kwargs
        self.rng = numpy.random.default_rng(seed=seed)

        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        while self.counter < self.pop_size:
            self.counter += 1
            a = self.rng.normal(self.mu_a, self.sigma_a)
            b = self.rng.normal(self.mu_b, self.sigma_b)

            return ExponentialForgetting(
                self.n_items,
                *self.args,
                **self.kwargs,
                seed=self.seed.spawn(1)[0],
                a=a,
                b=b,
            )
        raise StopIteration

    def __repr__(self):
        _str = "Exponential forgetting\n"
        _str += f"pop. size = {self.pop_size}\n"
        _str += f"a ~ Gaussian({self.mu_a:.3e}, {self.sigma_a**2:.3e}) \n"
        _str += f"b ~ Gaussian({self.mu_b:.3e}, {self.sigma_b**2:.3e}) \n"
        return _str


if __name__ == "__main__":
    # recalls = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    # ks = [i for i in range(len(recalls))]
    # deltats = [100, 100, 100, 1700, 200, 200, 1600, 100, 100, 3800]
    # a = -1.1
    # b = 0.6
    # ll = 0
    # for recall, k, deltat in zip(recalls, ks, deltats):
    #     res1 = ef_log_likelihood_sample(1, k, deltat, a, b)
    #     res2 = ef_log_likelihood_sample(0, k, deltat, a, b)
    #     ll += ef_log_likelihood_sample(recall, k, deltat, a, b)
    #     print(res1, res2)
    # print(ll)

    a = 1e-1
    b = 0.5
    deltat = 150
    k = 3

    # q1
    res = __sym_ef_dq1_dalpha_sample(a, b, k, deltat)
    res_bis = ef_dq1_dalpha_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_dq1_dbeta_sample(a, b, k, deltat)
    res_bis = ef_dq1_dbeta_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_ddq1_dalpha_dalpha_sample(a, b, k, deltat)
    res_bis = ef_ddq1_dalpha_dalpha_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_ddq1_dalpha_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq1_dalpha_dbeta_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_ddq1_dbeta_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq1_dbeta_dbeta_sample(a, b, k, deltat)
    print(res, res_bis)

    # q0
    res = __sym_ef_dq0_dalpha_sample(a, b, k, deltat)
    res_bis = ef_dq0_alpha_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_dq0_dbeta_sample(a, b, k, deltat)
    res_bis = ef_dq0_beta_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_ddq0_dalpha_dalpha_sample(a, b, k, deltat)
    res_bis = ef_ddq0_dalpha_dalpha_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_ddq0_dalpha_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq0_dalpha_dbeta_sample(a, b, k, deltat)
    print(res, res_bis)

    res = __sym_ef_ddq0_dbeta_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq0_dbeta_dbeta_sample(a, b, k, deltat)
    print(res, res_bis)
