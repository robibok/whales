# some parts are based on Lasagne

from __future__ import absolute_import
from bunch import Bunch

import theano.tensor as T
import theano
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from TheanoLib import utils

def get_p(param):
    if isinstance(param, Bunch):
        return param.param
    else:
        return param

def get_lr(param):
    if isinstance(param, Bunch):
        if ('lr' not in param.keys()) or param.lr is None:
            return 1.0
        else:
            return param.lr
    else:
        return 1.0


def make_some_noise(all_parameters, all_grads, eta=0.01, gamma=0.55, starting_time=0):
    iter_t_prev = theano.shared(np.asarray(starting_time, dtype=theano.config.floatX))
    iter_t = iter_t_prev + 1

    new_grads = []

    for param, grad in zip(all_parameters, all_grads):
        shape = get_p(param).get_value(borrow=True).shape
        std = T.sqrt(eta / ((1. + iter_t) ** gamma))
        noise = RandomStreams().normal(shape, avg=0., std=std, dtype=theano.config.floatX)

        if len(shape) == 1:
            grad = utils.PrintValueOp(0 * noise + grad, 'old_grad')

        new_grad = grad + noise

        if len(shape) == 1:
            new_grad = utils.PrintValueOp(new_grad, 'new_grad')

        new_grads.append(new_grad)

    t_update = (iter_t_prev, iter_t)

    return new_grads, t_update

# WARNING: Check correctenss
def gen_updates_rmsprop_and_nesterov_momentum(all_parameters, all_grads,
                                              learning_rate,
                                              rho=0.90, momentum=0.9,
                                              epsilon=1e-6, get_diffs=False, what_stats=None):
    updates = []
    infos = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(get_p(param_i).get_value() * 0.)
        rms_i = theano.shared(get_p(param_i).get_value() * 0.)

        rms_i_new = rho * rms_i + (1 - rho) * grad_i ** 2
        scaled_grad = grad_i / T.sqrt(rms_i_new + epsilon)
        step = -get_lr(param_i) * learning_rate * scaled_grad

        v = momentum * mparam_i + step  # new momemtum
        update = momentum * v + step
        w = get_p(param_i) + update  # new parameter values

        updates.append((rms_i, rms_i_new ))
        updates.append((mparam_i, v))
        updates.append((get_p(param_i), w))

        info = Bunch()
        info.grad = grad_i
        info.scaled_grad = scaled_grad

        if get_diffs:
            info.diff = update

        infos.append(info)

    return final_result(updates, all_parameters, infos, what_stats)


def gen_updates_rmsprop_by_graves(all_parameters, all_grads,
                                  learning_rate,
                                  rho=0.95, momentum=0.9,
                                  epsilon=0.0001,
                                  get_diffs=False,
                                  what_stats=None
                                  ):
    updates = []
    infos = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        delta_i = theano.shared(get_p(param_i).get_value() * 0.)
        n_i = theano.shared(get_p(param_i).get_value() * 0.)
        g_i = theano.shared(get_p(param_i).get_value() * 0.)

        n_i_new = rho * n_i + (1 - rho) * grad_i ** 2
        g_i_new = rho * g_i + (1 - rho) * grad_i
        delta_i_new = momentum * delta_i - get_lr(param_i) * learning_rate * grad_i / T.sqrt(n_i_new - (g_i_new ** 2) + epsilon)

        updates.append((n_i, n_i_new))
        updates.append((g_i, g_i_new))
        updates.append((delta_i, delta_i_new))
        update = delta_i_new
        updates.append((get_p(param_i), get_p(param_i) + update))

        info = Bunch()
        info.grad = grad_i

        if get_diffs:
            info.diff = update

        infos.append(info)

    return final_result(updates, all_parameters, infos, what_stats)


def gen_updates_adam(all_parameters, all_grads, learning_rate, beta_1=0.9, beta_2=0.999, eps=1e-8, get_diffs=False, what_stats=None):
    # WARN: not well tested, may contain bugs!!!
    updates = []
    infos = []

    t_prev = theano.shared(np.asarray(0., dtype=theano.config.floatX))

    t = t_prev + 1
    alpha_t = learning_rate * T.sqrt(1. - beta_2 ** t) / (1. - beta_1 ** t)


    for param_i, grad_i in zip(all_parameters, all_grads):
        m_i = theano.shared(get_p(param_i).get_value() * 0.)
        v_i = theano.shared(get_p(param_i).get_value() * 0.)

        m_i_new = beta_1 * m_i + (1 - beta_1) * grad_i
        v_i_new = beta_2 * v_i + (1 - beta_2) * T.sqr(grad_i)

        update = -get_lr(param_i) * alpha_t * m_i_new / (T.sqrt(v_i_new) + eps)

        updates.append((m_i, m_i_new))
        updates.append((v_i, v_i_new))
        updates.append((get_p(param_i), get_p(param_i) + update))

        info = Bunch()
        info.grad = grad_i

        if get_diffs:
            info.diff = update

        infos.append(info)


    updates.append((t_prev, t))

    return final_result(updates, all_parameters, infos, what_stats)



def gen_updates_rmsprop(all_parameters, all_grads, learning_rate=1.0, rho=0.9, epsilon=1e-6, get_diffs=False, what_stats=None):
    """
    epsilon is not included in Hinton's video, but to prevent problems with relus repeatedly having 0 gradients, it is included here.
    Watch this video for more info: http://www.youtube.com/watch?v=O3sxAc4hxZU (formula at 5:20)
    also check http://climin.readthedocs.org/en/latest/rmsprop.html
    """
    all_accumulators = [theano.shared(get_p(param_i).get_value() * 0.) for param_i in
                        all_parameters]  # initialise to zeroes with the right shape

    updates = []
    infos = []

    for param_i, grad_i, acc_i in zip(all_parameters, all_grads, all_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
        updates.append((acc_i, acc_i_new))
        scaled_grad = grad_i / T.sqrt(acc_i_new + epsilon)

        update = - get_lr(param_i) * learning_rate * scaled_grad
        updates.append((get_p(param_i), get_p(param_i) + update))

        info = Bunch()
        info.grad = grad_i
        info.scaled_grad = scaled_grad

        if get_diffs:
            info.diff = update
        infos.append(info)

    return final_result(updates, all_parameters, infos, what_stats)


def gen_updates_sgd(all_parameters, all_grads, learning_rate, get_diffs=False, what_stats=None):
    updates = []
    infos = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        update = - get_lr(param_i) * learning_rate * grad_i
        updates.append((get_p(param_i), get_p(param_i) + update))

        info = Bunch()
        info.grad = grad_i

        if get_diffs:
            info.diff = update

        infos.append(info)

    return final_result(updates, all_parameters, infos, what_stats)


def gen_updates_momentum(all_parameters, all_grads, learning_rate, momentum, get_diffs=False, what_stats=None):
    assert (momentum is not None)

    updates = []
    infos = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        velocity = theano.shared(np.zeros(get_p(param_i).get_value(borrow=True).shape, dtype=get_p(param_i).dtype),
                                 broadcastable=get_p(param_i).broadcastable)
        new_velocity = momentum * velocity - get_lr(param_i) * learning_rate * grad_i
        update = new_velocity
        updates.append((get_p(param_i), get_p(param_i) + update))
        updates.append((velocity, new_velocity))

        info = Bunch()
        info.grad = grad_i

        if get_diffs:
            info.diff = update

        infos.append(info)

    return final_result(updates, all_parameters, infos, what_stats)


def gen_updates_nesterov_momentum(all_parameters, all_grads, learning_rate, momentum, get_diffs=False, what_stats=None):
    assert (momentum is not None)
    updates = []
    infos = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        velocity = theano.shared(np.zeros(get_p(param_i).get_value(borrow=True).shape, dtype=get_p(param_i).dtype),
                                 broadcastable=get_p(param_i).broadcastable)
        # TODO: check the formulas
        # https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
        new_velocity = momentum * velocity - get_lr(param_i) * learning_rate * grad_i
        update = (momentum ** 2) * velocity - (1 + momentum) * get_lr(param_i) * learning_rate * grad_i
        updates.append((get_p(param_i), get_p(param_i) + update))
        updates.append((velocity, new_velocity))

        info = Bunch()
        info.grad = grad_i

        if get_diffs:
            info.diff = update

        infos.append(info)

    return final_result(updates, all_parameters, infos, what_stats)


def get_updates(all_parameters, all_grads, learning_rate, method, momentum=None, get_diffs=False, what_stats=None,
                add_noise=False, eta=0.01, gamma=0.55, starting_time=0):
    if add_noise:
        print 'adding noise'
        all_grads, t_update = make_some_noise(all_parameters, all_grads, eta=eta, gamma=gamma, starting_time=starting_time)

    if method == 'rmsprop':
        res = gen_updates_rmsprop(all_parameters=all_parameters,
                                  all_grads=all_grads,
                                  learning_rate=learning_rate,
                                  rho=0.9,
                                  get_diffs=get_diffs,
                                  what_stats=what_stats)
    elif method == 'rmsnest':
        res = gen_updates_rmsprop_and_nesterov_momentum(all_parameters=all_parameters,
                                                        all_grads=all_grads,
                                                        learning_rate=learning_rate,
                                                        rho=0.9,
                                                        momentum=0.9,
                                                        get_diffs=get_diffs,
                                                        what_stats=what_stats)
    elif method == 'rmsgraves':
        res = gen_updates_rmsprop_by_graves(
            all_parameters=all_parameters,
            all_grads=all_grads,
            learning_rate=learning_rate,
            get_diffs=get_diffs,
            what_stats=what_stats)

    elif method == 'sgd':
        res = gen_updates_sgd(all_parameters=all_parameters,
                              all_grads=all_grads,
                              learning_rate=learning_rate,
                              get_diffs=get_diffs,
                              what_stats=what_stats)
    elif method == 'momentum':
        res = gen_updates_momentum(all_parameters=all_parameters,
                                   all_grads=all_grads,
                                   learning_rate=learning_rate,
                                   momentum=momentum,
                                   get_diffs=get_diffs,
                                   what_stats=what_stats)
    elif method == 'nesterov_momentum':
        res = gen_updates_nesterov_momentum(all_parameters=all_parameters,
                                            all_grads=all_grads,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            get_diffs=get_diffs,
                                            what_stats=what_stats)
    elif method == 'adam':
        res = gen_updates_adam(all_parameters=all_parameters,
                               all_grads=all_grads,
                               learning_rate=learning_rate,
                               get_diffs=get_diffs,
                               what_stats=what_stats)

    else:
        raise RuntimeError('Unknown update method: ' + method)

    if add_noise:
        res[0].append(t_update)

    return res

def final_result(updates, params, infos, what_stats):
    from TheanoLib.monitor_stats import gen_stats
    return updates, gen_stats(params, infos, what_stats)

