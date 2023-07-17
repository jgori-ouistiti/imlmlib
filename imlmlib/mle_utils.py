import numpy
import scipy
import scipy.optimize as opti
import functools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


# class ConfidenceEllipsisNotDefinedError(Exception):
#     """ConfidenceEllipsisNotDefinedError

#     Raised when the estimated covariance matrix is singular
#     """

#     pass



def estim_mle_one_trial(
    times, recalls, k_vector, likelihood_function, optimizer_kwargs, guess, **kwargs
):
    # note: removed possibility to specify transform function
    # note: check the k_vector

    # invert sign and deal with argument shape
    ll_lambda = lambda guess, args: -likelihood_function(guess, *args)

    res = opti.minimize(
        ll_lambda, guess, args=[times, recalls, k_vector], **optimizer_kwargs
    )

    return res


numpy.seterr(divide="raise")


def _confidence_ellipsis(x, cov, critical_value = 5.991, **kwargs):
    # critical values can be looked up in a chisquared table with df = 2

    eigen_values, eigen_vectors = numpy.linalg.eig(cov)
    indexes = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[indexes]
    eigen_vectors = eigen_vectors[:,indexes]

    ellipsis_orientation = numpy.arctan2(eigen_vectors[:,0][1], eigen_vectors[:,0][0])
    
    ellipsis_large_axis = 2*numpy.sqrt(critical_value*eigen_values[0])
    ellipsis_small_axis = 2*numpy.sqrt(critical_value*eigen_values[1])
    return Ellipse(x, ellipsis_large_axis, ellipsis_small_axis, ellipsis_orientation, **kwargs)


def confidence_ellipse(inferred_parameters, estimated_covariance_matrix,  alpha_scale = 'log', confidence_levels = [.68, .95], plot = False, ax = None, colors = ['#B0E0E6', '#87CEEB'], CI95 = False, plot_kwargs = {'color': 'red', 'marker': 'D', 'label' : 'MLE'}):
    """estimated_covariance_matrix = numpy.linalg.inv(J)"""
    

    x = inferred_parameters
    if alpha_scale == 'log':
        estimated_covariance_matrix, CI_log_alpha, CI_beta = delta_method_log_CI(*x, estimated_covariance_matrix)
        x = (numpy.log10(x[0]), x[1])

    if plot is False:
        return CI_log_alpha, CI_beta
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
    critical_values = [scipy.stats.chi2.ppf(cl, 2) for cl in confidence_levels]
    for critical_value, color, cl in zip(critical_values[::-1], colors[::-1], confidence_levels[::-1]):
        ax.add_patch(_confidence_ellipsis(x, estimated_covariance_matrix, critical_value, fill = True, facecolor = color, edgecolor= 'b', label = f'Confidence level:{cl}'))

    ax.plot(*x, **plot_kwargs)
        

    if CI95:
        return ax, CI_log_alpha, CI_beta
    else:
        return ax



def identify_alpha_beta_from_recall_sequence(
    recall_sequence,
    deltas,
    guess=(1e-3, 0.5),
    optim_kwargs={"method": "L-BFGS-B", "bounds": [(1e-7, 5e-1), (0, 0.99)]},
    verbose=True,
    k_vector=None,
):

    infer_results = estim_mle_one_trial(
        deltas,
        recall_sequence,
        k_vector,
        ef_get_per_participant_likelihood_transform,
        optim_kwargs,
        guess,
    )

    if verbose:
        print(infer_results)

    return infer_results



def ef_get_sequence_observed_information_matrix(
    recall_sequence, deltas, alpha, beta, k_vector=None
):
    """ef_get_sequence_observed_information_matrix _summary_

    Returns the observed information matrix J.


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
    if k_vector is None:
        k_vector = list(range(1, len(deltas)))

    # print(recall_sequence, deltas, k_vector)
    for recall, delta, k in zip(recall_sequence, deltas, k_vector):
        # print(recall, delta, k)
        if recall == 1:
            J_11 += ef_ddq1_dalpha_dalpha_sample(alpha, beta, k, delta)
            J_12 += ef_ddq1_dalpha_dbeta_sample(alpha, beta, k, delta)
            J_22 += ef_ddq1_dbeta_dbeta_sample(alpha, beta, k, delta)
        elif recall == 0:
            J_11 += ef_ddq0_dalpha_dalpha_sample(alpha, beta, k, delta)
            J_12 += ef_ddq0_dalpha_dbeta_sample(alpha, beta, k, delta)
            J_22 += ef_ddq0_dbeta_dbeta_sample(alpha, beta, k, delta)
        else:
            raise ValueError(f"recall is not either 1 or 0, but is {recall}")

    J = -numpy.array([[J_11, J_12], [J_12, J_22]])
    return J, len(deltas)


def delta_method_log_CI(alpha, beta, var):
    """var should be inverse of J observed information matrix
    """
    grad = numpy.array([[1 / alpha / numpy.log(10), 0], [0, 1]])
    new_var = grad.T @ var @ grad
    CI_alpha_low, CI_alpha_high, CI_beta_low, CI_beta_high = _CI_asymptotical(
        numpy.log10(alpha), beta, new_var[0,0], new_var[1,1]
    )

    return new_var, (CI_alpha_low, CI_alpha_high), (CI_beta_low, CI_beta_high)


def _CI_asymptotical(alpha, beta, inv_J_00, inv_J_11, critical_value=1.96):
    with numpy.errstate(invalid="raise"):
        try:
            CI_alpha_low = alpha - critical_value * numpy.sqrt(inv_J_00)
            CI_alpha_high = alpha + critical_value * numpy.sqrt(inv_J_00)
        except FloatingPointError:
            CI_alpha_low = numpy.nan
            CI_alpha_high = numpy.nan

        try:
            CI_beta_low = beta - critical_value * numpy.sqrt(inv_J_11)
            CI_beta_high = beta + critical_value * numpy.sqrt(inv_J_11)
        except FloatingPointError:
            CI_beta_low = numpy.nan
            CI_beta_high = numpy.nan

    return CI_alpha_low, CI_alpha_high, CI_beta_low, CI_beta_high


def observed_information_matrix(recall_sequence, deltas, alpha, beta, k_vector=None):
    J, n = ef_get_sequence_observed_information_matrix(
        recall_sequence, deltas, alpha, beta, k_vector=k_vector
    )
    return J

def get_confidence_single(
    J, level=0.05, verbose=True
):
    

    inv_J = numpy.linalg.inv(J)

    if level != 0.05:
        raise NotImplementedError
    CI_alpha_low, CI_alpha_high, CI_beta_low, CI_beta_high = _CI_asymptotical(
        alpha, beta, inv_J[0, 0], inv_J[1, 1], critical_value=1.96
    )

    if verbose:
        print(f"N observations: {n}")
        print(f"Observed Information Matrix: {inv_J}")
        print("Asymptotic confidence intervals (only valid for large N)")
        print(f"alpha: [{CI_alpha_low:.3e}, {CI_alpha_high:.3e}]")
        print(f"beta: [{CI_beta_low:.3e}, {CI_beta_high:.3e}]")

    return J, (CI_alpha_low, CI_alpha_high), (CI_beta_low, CI_beta_high)

