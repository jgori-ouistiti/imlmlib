import numpy
import scipy.optimize as opti
import functools
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from tqdm import tqdm
from itertools import chain

from imlmlib.mem_utils import experiment
from imlmlib.exponential_forgetting import (
    ef_get_per_participant_likelihood_transform,
    ef_ddq1_dalpha_dalpha_sample,
    ef_ddq1_dalpha_dbeta_sample,
    ef_ddq1_dbeta_dbeta_sample,
    ef_ddq0_dalpha_dalpha_sample,
    ef_ddq0_dalpha_dbeta_sample,
    ef_ddq0_dbeta_dbeta_sample,
)


def grid_ll(likelihood, schedule, xp_data, discretize, verbose=False):
    """grid_ll _summary_

    import matplotlib.pyplot as plt

    discretize = (40, 10)
    res = grid_ll(ef_get_global_likelihood, schedule_one, data, discretize)
    ll = numpy.clip(res[:, 2], max(res[:, 2]) * 2, max(res[:, 2])).reshape(*discretize)
    maxll = res[numpy.argmax(ll), :2]
    plt.imshow(ll)
    plt.plot(maxll[1] * 10, 29, "rD")
    plt.plot(0.5 * 10, (-2 + 5) * 10)
    plt.show()
    exit()
    """
    # invert sign and deal with argument shape
    ll_lambda = lambda guess, args: -likelihood(guess, *args)

    data = xp_data[0]

    grid = numpy.dstack(
        numpy.meshgrid(
            numpy.linspace(-5, -1, discretize[0]),
            numpy.linspace(0, 0.99, discretize[1]),
        )
    ).reshape(-1, 2)

    ll = numpy.zeros((grid.shape[0], 3))
    for i, theta in enumerate(grid):
        ll[i, :2] = theta
        ll[i, 2] = -ll_lambda(theta, [data, schedule])

    return ll


def estim_mle_one_trial(
    times,
    recalls,
    transform,
    k_vector,
    likelihood_function,
    optimizer_kwargs,
    guess,
):
    # invert sign and deal with argument shape
    ll_lambda = lambda guess, args: -likelihood_function(guess, *args)

    res = opti.minimize(
        ll_lambda, guess, args=[times, recalls, transform, k_vector], **optimizer_kwargs
    )

    return res


def confidence_ellipse(x, y, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """straightforward adaptation from :
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    cov : array-like, shape (n, n)
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = numpy.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = numpy.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(x, y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def identify_alpha_beta_from_recall_sequence(
    recall_sequence,
    deltas,
    guess=(1e-3, 0.5),
    optim_kwargs={"method": "L-BFGS-B", "bounds": [(1e-7, 5e-1), (0, 0.99)]},
    verbose=True,
    k_vector=None,
):

    # for r, d, k in zip(recall_sequence, deltas, k_vector):
    #     print(r, d, k)

    infer_results = estim_mle_one_trial(
        deltas,
        recall_sequence,
        lambda a, b: (a, b),
        k_vector,
        ef_get_per_participant_likelihood_transform,
        optim_kwargs,
        guess,
    )

    if verbose:
        print(infer_results)

    I, _, CI_alpha, CI_beta = get_confidence_single(
        recall_sequence, deltas, *infer_results.x, verbose=verbose
    )
    return infer_results, I, CI_alpha, CI_beta


def sample_mle(REPL, Nseq, population_model, schedule):
    data = experiment(population_model, schedule, replications=REPL * Nseq)
    results = numpy.zeros((8, REPL))
    for r in tqdm(range(REPL)):
        recall_sequence = data[r * Nseq : (r + 1) * Nseq, 0, :, 0]
        recall = recall_sequence[:, 1:].ravel(order="C")
        k_vector = list(
            chain(*[list(range(recall_sequence.shape[1] - 1)) for u in range(Nseq)])
        )
        deltas = list(chain(*[numpy.diff(schedule.times) for u in range(Nseq)]))
        optim_kwargs = {"method": "L-BFGS-B", "bounds": [(1e-5, 1e-1), (0, 0.99)]}
        verbose = False
        guess = (1e-3, 0.5)
        (
            inference_results,
            I,
            CI_alpha,
            CI_beta,
        ) = identify_alpha_beta_from_recall_sequence(
            recall,
            deltas,
            optim_kwargs=optim_kwargs,
            verbose=verbose,
            guess=guess,
            k_vector=k_vector,
        )
        results[:6, r] = *inference_results.x, *CI_alpha, *CI_beta
        new_I, CI_alpha, CI_beta = delta_method_log_CI(*inference_results.x, I)
        results[6:, r] = (*CI_alpha,)
    return results, new_I


def ef_get_sequence_observed_information_matrix(recall_sequence, deltas, alpha, beta):
    """ef_get_sequence_observed_information_matrix _summary_

    Returns the observed information matrix J divided by N the number of observations.
    Asymptotically, J/N converges to the fischer information matrix, which we can use to create confidence intervals.

    :param recall_sequence: _description_
    :type recall_sequence: _type_
    :param time_sequence: _description_
    :type time_sequence: _type_
    :param alpha: _description_
    :type alpha: _type_
    :param beta: _description_
    :type beta: _type_
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    J_11 = 0
    J_12 = 0
    J_22 = 0
    for n, (recall, delta) in enumerate(zip(recall_sequence, deltas)):
        if recall == 1:
            J_11 += ef_ddq1_dalpha_dalpha_sample(alpha, beta, n, delta)
            J_12 += ef_ddq1_dalpha_dbeta_sample(alpha, beta, n, delta)
            J_22 += ef_ddq1_dbeta_dbeta_sample(alpha, beta, n, delta)
        elif recall == 0:
            J_11 += ef_ddq0_dalpha_dalpha_sample(alpha, beta, n, delta)
            J_12 += ef_ddq0_dalpha_dbeta_sample(alpha, beta, n, delta)
            J_22 += ef_ddq0_dbeta_dbeta_sample(alpha, beta, n, delta)
        else:
            raise ValueError(f"recall is not either 1 or 0, but is {recall}")

    J = -numpy.array([[J_11, J_12], [J_12, J_22]])
    return numpy.linalg.inv(J), J, n


def delta_method_log_CI(alpha, beta, var):
    grad = numpy.array([[1 / alpha / numpy.log(10), 0], [0, 1]])

    new_var = grad.T @ var @ grad
    CI_alpha_low, CI_alpha_high, CI_beta_low, CI_beta_high = _CI_asymptotical(
        numpy.log10(alpha), beta, new_var
    )

    return new_var, (CI_alpha_low, CI_alpha_high), (CI_beta_low, CI_beta_high)


def _CI_asymptotical(alpha, beta, var):
    with numpy.errstate(invalid="raise"):
        try:
            CI_alpha_low = alpha - 1.96 * numpy.sqrt(var[0, 0])
            CI_alpha_high = alpha + 1.96 * numpy.sqrt(var[0, 0])
        except FloatingPointError:
            CI_alpha_low = numpy.nan
            CI_alpha_high = numpy.nan
        try:
            CI_beta_low = beta - 1.96 * numpy.sqrt(var[1, 1])
            CI_beta_high = beta + 1.96 * numpy.sqrt(var[1, 1])
        except FloatingPointError:
            CI_beta_low = numpy.nan
            CI_beta_high = numpy.nan
    return CI_alpha_low, CI_alpha_high, CI_beta_low, CI_beta_high


def get_confidence_single(recall_sequence, deltas, alpha, beta, verbose=True):

    I, J, n = ef_get_sequence_observed_information_matrix(
        recall_sequence, deltas, alpha, beta
    )

    CI_alpha_low, CI_alpha_high, CI_beta_low, CI_beta_high = _CI_asymptotical(
        alpha, beta, I
    )

    if verbose:
        print(f"N observations: {n}")
        print(f"Observed Information Matrix: {I}")
        print("Asymptotic confidence intervals (only valid for large N)")
        print(f"alpha: [{CI_alpha_low:.3e}, {CI_alpha_high:.3e}]")
        print(f"beta: [{CI_beta_low:.3e}, {CI_beta_high:.3e}]")

    return I, J, (CI_alpha_low, CI_alpha_high), (CI_beta_low, CI_beta_high)
