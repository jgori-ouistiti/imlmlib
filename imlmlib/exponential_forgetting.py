# import warnings
# warnings.filterwarnings("error")
# https://stackoverflow.com/questions/63979540/python-how-to-filter-specific-warning  --> see this link for filtering
import numpy


from imlmlib.mem_utils import MemoryModel
import seaborn
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas
import statsmodels.api as sm
import statsmodels.formula.api as smf


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
    # return __sym_ef_ddq1_dalpha_dalpha_sample(alpha, beta, k, deltat)
    return 0


def ef_ddq1_dalpha_dbeta_sample(alpha, beta, k, deltat):
    # return __sym_ef_ddq1_dalpha_dbeta_sample(alpha, beta, k, deltat)
    return k * (1 - beta) ** (k - 1) * deltat


def ef_ddq1_dbeta_dbeta_sample(alpha, beta, k, deltat):
    # return __sym_ef_ddq1_dbeta_dbeta_sample(alpha, beta, k, deltat)
    return -alpha * k * (k - 1) * (1 - beta) ** (k - 2) * deltat


def ef_ddq0_dalpha_dalpha_sample(alpha, beta, k, deltat):
    # return __sym_ef_ddq0_dalpha_dalpha_sample(alpha, beta, k, deltat)
    if deltat == numpy.inf:
        return 0
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
    # return __sym_ef_ddq0_dalpha_dbeta_sample(alpha, beta, k, deltat)
    if deltat == numpy.inf:
        return 0
    with numpy.errstate(divide="raise", invalid="raise"):
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
    # return __sym_ef_ddq0_dbeta_dbeta_sample(alpha, beta, k, deltat)
    if deltat == numpy.inf:
        return 0
    with numpy.errstate(divide="raise", invalid="raise"):
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
        with numpy.errstate(over="raise", invalid="raise"):
            try:
                exp = numpy.exp(-a * (1 - b) ** k * deltat)

                # exp = numpy.clip(exp, a_min=0, a_max=1 - 1e-4)
                return numpy.log(1 - exp)
            except FloatingPointError:
                return -1e6

    else:
        raise ValueError(f"Recall is not 0 or 1, but is {recall}")


# Old version, with recalls[:1]
#
# def ef_get_per_participant_likelihood_transform(
#     theta, deltas, recalls, transform, k_vector=None
# ):
#     ll = 0
#     alpha, beta = theta

#     if k_vector is None:
#         for nsched, recall in enumerate(recalls[1:]):
#             ll += ef_log_likelihood_sample(
#                 recall, nsched, deltas[nsched], alpha, beta, transform
#             )
#     else:
#         for n, (k, recall) in enumerate(zip(k_vector, recalls)):
#             ll += ef_log_likelihood_sample(recall, k, deltas[n], alpha, beta, transform)
#     return ll


def ef_get_per_participant_likelihood_transform(
    theta,
    deltas,
    recalls,
    k_vector=None,
    transform=None,
):
    """ef_get_per_participant_likelihood_transform

    .. warning::

        if k_vector is None, then it is expected that you give the first trial as well ie the first one where recall is impossible.

    :param theta: _description_
    :type theta: _type_
    :param deltas: _description_
    :type deltas: _type_
    :param recalls: _description_
    :type recalls: _type_
    :param transform: _description_, defaults to None
    :type transform: _type_, optional
    :param k_vector: _description_, defaults to None
    :type k_vector: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    ll = 0
    alpha, beta = theta
    if transform is None:
        transform = lambda a, b: (a, b)
    if k_vector is None:
        for nsched, recall in enumerate(recalls):
            dll = ef_log_likelihood_sample(
                recall, nsched, deltas[nsched], alpha, beta, transform
            )
            ll += dll
    else:
        for n, (k, recall) in enumerate(zip(k_vector, recalls)):
            dll = ef_log_likelihood_sample(recall, k, deltas[n], alpha, beta, transform)
            ll += dll
    return ll


class ExponentialForgetting(MemoryModel):
    def __init__(self, nitems, *args, a=0.1, b=0.5, **kwargs):
        super().__init__(nitems, *args, **kwargs)
        self.a, self.b = a, b
        self.reset()

    def reset(self, *args, a=None, b=None):
        super().reset()
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

        if len(args) == 0:
            self.counters = numpy.zeros((self.nitems, 2))
            self.counters[:, 1] = -numpy.inf
        else:
            self.counters = numpy.zeros((self.n_items, 2))
            self.counters[:, 0] = args[0]
            self.counters[:, 1] = args[1]

    def update(self, item, time, N=None):
        item = int(item)
        if N is None:
            self.counters[item, 0] += 1
        else:
            self.counters[item, 0] = N
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
        population_size=1,
        n_items=1,
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
        self.pop_size = population_size
        self.seed = numpy.random.SeedSequence(seed)
        self.n_items = n_items
        self.kwargs = kwargs
        self.rng = numpy.random.default_rng(seed=self.seed)

        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.pop_size:
            self.counter += 1
            if self.sigma_a is not None:
                a = self.rng.normal(self.mu_a, self.sigma_a)
            else:
                a = self.mu_a
            if self.sigma_b is not None:
                b = self.rng.normal(self.mu_b, self.sigma_b)
            else:
                b = self.mu_b

            return ExponentialForgetting(
                self.n_items,
                **self.kwargs,
                seed=self.seed.spawn(1)[0],
                a=a,
                b=b,
            )
        raise StopIteration

    def __repr__(self):
        _str = "Exponential forgetting\n"
        _str += f"pop. size = {self.pop_size}\n"
        if self.sigma_a is not None:
            _str += f"a ~ Gaussian({self.mu_a:.3e}, {self.sigma_a**2:.3e}) \n"
        else:
            _str += f"a ~ Gaussian({self.mu_a:.3e}, 0) \n"
        if self.sigma_b is not None:
            _str += f"b ~ Gaussian({self.mu_b:.3e}, {self.sigma_b**2:.3e}) \n"
        else:
            _str += f"b ~ Gaussian({self.mu_b:.3e}, 0) \n"
        return _str


#### ============================= Plot

from seaborn.regression import _RegressionPlotter

### Patching the binned regression results from seaborn to access to the binned scatterplot data (if needed).


class _PatchedRegressionPlotter(_RegressionPlotter):
    def scatterplot(self, ax, kws):
        """Draw the data."""
        # Treat the line-based markers specially, explicitly setting larger
        # linewidth than is provided by the seaborn style defaults.
        # This would ideally be handled better in matplotlib (i.e., distinguish
        # between edgewidth for solid glyphs and linewidth for line glyphs
        # but this should do for now.
        line_markers = ["1", "2", "3", "4", "+", "x", "|", "_"]
        if self.x_estimator is None:
            if "marker" in kws and kws["marker"] in line_markers:
                lw = mpl.rcParams["lines.linewidth"]
            else:
                lw = mpl.rcParams["lines.markeredgewidth"]
            kws.setdefault("linewidths", lw)

            if not hasattr(kws["color"], "shape") or kws["color"].shape[1] < 4:
                kws.setdefault("alpha", 0.8)

            x, y = self.scatter_data
            ax.scatter(x, y, **kws)
        else:
            # TODO abstraction
            ci_kws = {"color": kws["color"]}
            if "alpha" in kws:
                ci_kws["alpha"] = kws["alpha"]
            ci_kws["linewidth"] = mpl.rcParams["lines.linewidth"] * 1.75
            kws.setdefault("s", 50)

            xs, ys, cis = self.estimate_data
            if [ci for ci in cis if ci is not None]:
                for x, ci in zip(xs, cis):
                    ax.plot([x, x], ci, **ci_kws)
            ax.scatter(xs, ys, **kws)
            self.xs = xs
            self.ys = ys
            self.cis = cis


def regplot(
    data=None,
    *,
    x=None,
    y=None,
    x_estimator=None,
    x_bins=None,
    x_ci="ci",
    scatter=True,
    fit_reg=True,
    ci=95,
    n_boot=1000,
    units=None,
    seed=None,
    order=1,
    logistic=False,
    lowess=False,
    robust=False,
    logx=False,
    x_partial=None,
    y_partial=None,
    truncate=True,
    dropna=True,
    x_jitter=None,
    y_jitter=None,
    label=None,
    color=None,
    marker="o",
    scatter_kws=None,
    line_kws=None,
    ax=None,
):
    """
    # regplot.xs, regplot.ys, regplot.cis --> summary data

    """
    plotter = _PatchedRegressionPlotter(
        x,
        y,
        data,
        x_estimator,
        x_bins,
        x_ci,
        scatter,
        fit_reg,
        ci,
        n_boot,
        units,
        seed,
        order,
        logistic,
        lowess,
        robust,
        logx,
        x_partial,
        y_partial,
        truncate,
        dropna,
        x_jitter,
        y_jitter,
        color,
        label,
    )

    if ax is None:
        ax = plt.gca()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax, plotter


def plot_exponent_scatter(exponent, recall, ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    _, _regplot = regplot(
        x=exponent,
        y=recall,
        ax=ax,
        fit_reg=False,
        x_bins=40,
        label="Estimated recall probability",
    )
    _, _ = regplot(
        x=exponent, y=recall, ax=ax, fit_reg=False, label="Recall events", marker="."
    )
    _x = numpy.linspace(numpy.min(exponent), numpy.max(exponent), 100)
    ax.plot(_x, [numpy.exp(x) for x in _x], "-", label="exponential forgetting model")
    ax.set_xlabel(r"$\mathrm{exponent~}\omega$")
    ax.set_ylabel("Recall (events and probabilities)")
    return ax, _regplot


def loglogpplot(k_repetition, recall, deltas):
    sequence = [(k, r, d) for k, r, d in zip(k_repetition, recall, deltas)]
    df = pandas.DataFrame(sequence)
    df.columns = ["repetition", "recall", "deltat"]
    df = df[df["repetition"] >= 0]
    df["discretized_deltas"] = pandas.cut(df["deltat"], bins=15, labels=False)
    df_group = df.groupby(["repetition", "discretized_deltas"]).mean()
    df_group["log_delta"] = numpy.log(df_group["deltat"])
    df_group = df_group[df_group["recall"] != 0]
    df_group["minuslogp"] = -numpy.log(df_group["recall"])
    df_group = df_group[df_group["minuslogp"] != 0]
    df_group["logminuslogp"] = numpy.log(df_group["minuslogp"])

    df_group = df_group.reset_index()
    model = smf.mixedlm(
        "logminuslogp ~ log_delta", df_group, groups=df_group["repetition"]
    )
    _fit = model.fit()
    ri = [v[0] for v in _fit.random_effects.values()]
    beta_estim = 1 - numpy.exp(numpy.mean(numpy.diff(ri)))
    alpha_estim = numpy.exp(_fit.params["Intercept"] + ri[0])
    estim = {"alpha_estim": alpha_estim, "beta_estim": beta_estim}

    fg = seaborn.lmplot(
        x="log_delta",
        y="logminuslogp",
        data=df_group.reset_index(),
        hue="repetition",
    )
    ax = fg.axes.flatten()[0]
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, "r-", lw=2, label="y=x")
    ax.text(numpy.mean(xlim) + 0.2, numpy.mean(xlim) - 0.2, "y=x", color="r")
    fg.set_xlabels(r"$\log(\Delta_t)$")
    fg.set_ylabels(r"$\log(-\log p)$")
    return fg, ax, estim


def diagnostics(alpha, beta, k_repetition, deltas, recall):
    exponent = [
        -alpha * (1 - beta) ** (k) * dt for (k, dt) in zip(k_repetition, deltas)
    ]
    fig, axs = plt.subplots(nrows=1, ncols=1)
    ax, regplot = plot_exponent_scatter(exponent, recall, ax=axs)
    ax.legend()
    fg, ax, estim = loglogpplot(k_repetition, recall, deltas)

    return fig, (fg, ax, estim)


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
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_dq1_dbeta_sample(a, b, k, deltat)
    res_bis = ef_dq1_dbeta_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_ddq1_dalpha_dalpha_sample(a, b, k, deltat)
    res_bis = ef_ddq1_dalpha_dalpha_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_ddq1_dalpha_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq1_dalpha_dbeta_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_ddq1_dbeta_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq1_dbeta_dbeta_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    # q0
    res = __sym_ef_dq0_dalpha_sample(a, b, k, deltat)
    res_bis = ef_dq0_alpha_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_dq0_dbeta_sample(a, b, k, deltat)
    res_bis = ef_dq0_beta_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_ddq0_dalpha_dalpha_sample(a, b, k, deltat)
    res_bis = ef_ddq0_dalpha_dalpha_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_ddq0_dalpha_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq0_dalpha_dbeta_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    res = __sym_ef_ddq0_dbeta_dbeta_sample(a, b, k, deltat)
    res_bis = ef_ddq0_dbeta_dbeta_sample(a, b, k, deltat)
    assert numpy.abs(res - res_bis) < 1e-6

    ###################
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
    ]  # small schedule
    items = [0 for i in times]  # only one item
    ef = ExponentialForgetting(1, a=a, b=b)
    ef.reset()
    print(ef.counters)
    for n, (item, time) in enumerate(zip(items, times)):
        print("=====")
        print(n)
        ef.query_item(item, time)
        ef.update(item, time)
        print(ef.counters)
        # print(ef.compute_probabilities())
