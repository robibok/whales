from __future__ import absolute_import
from collections import OrderedDict
from bunch import Bunch
import theano


def function(inputs, outputs, *args, **kwargs):
    outputs_flat = []

    def walk(act, fun):
        if isinstance(act, Bunch):
            return Bunch({k: walk(v, fun) for (k, v) in act.iteritems()})
        elif isinstance(act, dict):
            return {k: walk(v, fun) for (k, v) in act.iteritems()}
        elif isinstance(act, OrderedDict):
            return {k: walk(v, fun) for (k, v) in act.iteritems()}
        elif isinstance(act, list):
            return map(lambda a: walk(a, fun), act)
        elif isinstance(act, str):
            return act
        else:
            return fun(act)

    def f(act):
        outputs_flat.append(act)
        return len(outputs_flat) - 1

    annotated = walk(outputs, f)

    theano_function = theano.function(inputs=inputs, outputs=outputs_flat, *args, **kwargs)

    def ret_function(*args):
        res_flat = theano_function(*args)

        def g(act):
            if isinstance(act, int):
                return res_flat[act]
            else:
                raise RuntimeError('Got ' + type(act) + ' should be int')

        ret_structured = walk(annotated, g)
        return ret_structured

    return ret_function

