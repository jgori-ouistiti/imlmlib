import numpy
import scipy.optimize as opti


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
    likelihood_function,
    optimizer_kwargs,
    guess,
    verbose=False,
):
    print(times)
    print(recalls)
    # invert sign and deal with argument shape
    ll_lambda = lambda guess, args: -likelihood_function(guess, *args)

    res = opti.minimize(ll_lambda, guess, args=[times, recalls], **optimizer_kwargs)

    return res


def estim_mle_N_trials(
    times,
    recalls,
    likelihood_function,
    optimizer_kwargs,
    guess,
    verbose=False,
):

    # recall.shape = (len(times), N)
    N = recalls.shape[1]

    if len(guess) == 1:
        guess = [guess for i in range(N)]
    else:
        if len(guess) != N:
            raise ValueError(
                f"guess should have length 1 or {N}, but guess is length {len(guess)}"
            )

    if len(times) == 1:
        times = [times[0] for i in range(N)]
    else:
        if len(times) != N:
            raise ValueError(
                f"times should have length 1 or {N}, but times is length {len(times)}"
            )

    N_parameter = len(guess[0])

    results = numpy.zeros((N, 2 * N_parameter))
    recalls = recalls.transpose(1, 0)
    for n, (recall, time, guess) in enumerate(zip(recalls, times, guess)):
        # invert sign and deal with argument shape
        results[n, :N_parameter] = guess
        results[n, N_parameter:] = estim_mle_one_trial(
            time, recall, likelihood_function, optimizer_kwargs, guess
        ).x

    return results


# def estim_mle(
#     schedule,
#     xp_data,
#     optimizer_kwargs,
#     guess,
#     verbose=False,
# ):

#     # invert sign and deal with argument shape
#     ll_lambda = lambda guess, args: -ef_get_global_likelihood(guess, *args)

#     iters = xp_data.shape[0]

#     infer_results = numpy.zeros((iters, 2 * len(guess)))
#     for i, data in enumerate(xp_data):
#         if verbose:
#             print(i)
#         infer_results[i, :2] = guess

#         res = opti.minimize(ll_lambda, guess, args=[data, schedule], **optimizer_kwargs)
#         infer_results[i, 2:] = res.x

#     return infer_results
