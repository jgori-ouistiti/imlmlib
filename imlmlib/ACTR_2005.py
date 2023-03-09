import numpy
import warnings

# make sure to catch numerical problems, that could make MLE unsuccessful
warnings.filterwarnings("error")


### ACT-R à la Pavlik Jr, P. I., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory: An activation‐based model of the spacing effect. Cognitive science, 29(4), 559-586.


# class ACTR_Pavlik2005_MLE(mem_utils.ACTR_Pavlik2005):


# def actr_log_likelihood_sample(recall, history, a, c, tau, s):

#     # rescaling value to linear

#     activation, _ = compute_activation_delay(c, a, history)
#     X = (tau - activation) / s
#     expfactor = numpy.exp(X)
#     logoneplusex = numpy.log(1 + expfactor)

#     if recall == 1:  # Warning: passing to array converts recall to float
#         return -logoneplusex
#     else:
#         return X - logoneplusex


if __name__ == "__main__":
    recalls = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ks = [i for i in range(len(recalls))]
    deltats = [100, 100, 100, 1700, 200, 200, 1600, 100, 100, 3800]
    a = -1.1
    b = 0.6
    ll = 0
    for recall, k, deltat in zip(recalls, ks, deltats):
        res1 = ef_log_likelihood_sample(1, k, deltat, a, b)
        res2 = ef_log_likelihood_sample(0, k, deltat, a, b)
        ll += ef_log_likelihood_sample(recall, k, deltat, a, b)
        print(res1, res2)
    print(ll)
