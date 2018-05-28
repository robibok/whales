from __future__ import absolute_import
from tabulate import tabulate
import theano.tensor as T

# PARAMS_SQR_MEAN = 0
# PARAMS_SQR_MAX = 1
#
# GRADS_SQR_MEAN = 2
# GRADS_SQR_MAX = 3
from TheanoLib import utils
from TheanoLib.optimization import get_p

AUX_NAMES = ['AUX_ABS_MEAN',
 'AUX_ABS_MAX']


class Stat(object):
    def comp_one(self, param, info):
        raise NotImplementedError()

    def comp_all(self, params, infos):
        res = []
        for param, info in zip(params, infos):
            res.append(self.comp_one(param, info))
        return res


def LpKrzynowekStat(p=2):
    class LpKrzynowekStat(Stat):
        NAME = 'L{} KRZYNOWEK'.format(p)
        def comp_one(self, param, info):
            return info.diff.norm(L=p) / get_p(param).norm(L=p)
    return LpKrzynowekStat()


class KrzynowekStat(Stat):
    NAME = 'KRZYNOWEK'
    def comp_one(self, param, info):
        return T.mean(T.abs_(info.diff / get_p(param)))


class ParamsAbsMean(Stat):
    NAME = 'PARAMS_ABS_MEAN'
    def comp_one(self, param, info):
        return T.mean(T.abs_(get_p(param)))


class ParamsAbsMax(Stat):
    NAME = 'PARAMS_ABS_MAX'
    def comp_one(self, param, info):
        return T.max(T.abs_(get_p(param)))


class GradsAbsMean(Stat):
    NAME = 'GRADS_ABS_MEAN'
    def comp_one(self, param, info):
        return T.mean(T.abs_(info.grad))


class GradsAbsMax(Stat):
    NAME = 'GRADS_ABS_MAX'
    def comp_one(self, param, info):
        return T.max(T.abs_(info.grad))


class DiffsL2(Stat):
    NAME = 'DIFFS_L2'
    def comp_one(self, param, info):
        return utils.L2(info.diff)


class DiffsAbsMax(Stat):
    NAME = 'DIFFS_ABS_MAX'
    def comp_one(self, param, info):
        return T.max(T.abs_(info.diff))

class ScaledGradAbsMean(Stat):
    NAME = 'SCALED_GRAD_ABS_MEAN'
    def comp_one(self, param, info):
        return T.mean(T.abs_(info.scaled_grad))



# class MonitorStats(object):
#     def __init__(self):
#         pass
#

# TODO: automatic graphs for all these generated values?
def gen_stats(params, infos, what_stats):
    if not what_stats:
        return []
    results = []
    for stat in what_stats:
        print len(params), len(infos)
        res = stat.comp_all(params, infos)
        print stat
        print res
        results.append(T.stack(*res))
    return T.stack(*results)


class AuxParamsAbsMean(Stat):
    NAME = 'AUX_PARAMS_ABS_MEAN'
    def comp_one(self, param, grad=None, diff=None):
        return T.mean(T.abs_(get_p(param)))


class AuxParamsAbsMax(Stat):
    NAME = 'AUX_PARAMS_ABS_MAX'
    def comp_one(self, param, grad=None, diff=None):
        return T.max(T.abs_(get_p(param)))


def gen_aux_stats(params, what_aux_stats):
    results = []
    for stat in what_aux_stats:
        res = stat.comp_all(params, None, None)
        results.append(T.stack(*res))
    return T.stack(*results)


def analyze_results(params, stats, what_stats, aux_names=None, aux_stats=None):
    d = {}

    for idx, what_stat in enumerate(what_stats):
        for idx2, param in enumerate(params):
            d[(what_stat, get_p(param))] = stats[idx][idx2]

    headers = ['+'] + [what_stat.NAME for what_stat in what_stats]
    table = []
    for param in params:
        row = [get_p(param).name]
        for what_stat in what_stats:
            row.append(str(d[(what_stat, get_p(param))]))
        table.append(row)

    print tabulate(table, headers, tablefmt='simple')

    ##################################################

    if aux_names is not None:
        print ''
        table2 = []
        for idx, name in enumerate(AUX_NAMES):
            table2.append([name] + aux_stats[idx].tolist())

        headers = ['+'] + aux_names
        print tabulate(table2, headers, tablefmt='simple')

