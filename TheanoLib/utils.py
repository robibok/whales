import cPickle
import os
import errno
import itertools
import termios
import sys

from sklearn import metrics

from bunch import Bunch
import matplotlib
from sklearn.metrics import auc

import ml_utils

if 'LOCATION' in os.environ and os.environ['LOCATION'] == 'aws':
    matplotlib.use('svg')

import theano
import numpy as np
import theano.tensor as T
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4,
                    threshold=30000,
                    linewidth=150)

################################################################


################################################################
def line(n=60):
    return n * '-'

def l2_sum(w):
    return (w ** 2).sum()


def l2_max(w):
    return T.max(w ** 2)


def l1_max(w):
    return T.max(T.abs_(w))


def l2_mean(w):
    return T.mean(w ** 2)


def write_summary_of_params(params):
    nof_params = 0
    print line()
    for p in params:
        a = np.prod(p.get_value().shape)
        print p.name, p.get_value().shape, a
        nof_params += a
    print 'nof_params', nof_params
    print line()


def get_l2_reg(params):
    sqr_mean = []
    for param in params:
        sqr_mean.append(T.mean(T.sqr(param)))

    return T.mean(T.stack(*sqr_mean))


def set_theano_fast_compile():
    theano.config.mode = 'FAST_COMPILE'


def set_theano_fast_run():
    theano.config.mode = 'FAST_RUN'


def set_theano_debug():
    theano.config.mode = 'DebugMode'


def theano_compilation_mode():
    return theano.config.mode


def mark(tensor):
    tensor.name = 'marked'


def PrintShapeOp(tensor, tensor_name):
    if ml_utils.DEBUG:
        return theano.printing.Print(tensor_name, attrs=('shape',))(tensor)
    else:
        return tensor


def PrintValueOp(tensor, tensor_name):
    if ml_utils.DEBUG:
        return theano.printing.Print(tensor_name, attrs=('__str__',))(tensor)
    else:
        return tensor


def tensor_desc_str(tensor, name=None):
    w = str(tensor.type) + ' broadcast = ' + str(tensor.type.broadcastable)
    if name is not None:
        return name + ' = ' + w
    else:
        return w


def tensor_desc(tensor, name=None):
    print tensor_desc_str(tensor, name)


def L2(t):
    return T.sqrt(T.sum(t ** 2))


def sum_from_list(l):
    r = l[0]
    for x in l:
        r = r + x
    return r


def max_from_list(l):
    r = l[0]
    for x in l:
        r = T.maximum(r, x)
    return r


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

def serialize(obj, path):
    f = file(path, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def deserialize(path):
    return cPickle.load(file(path, 'rb'))


def dump_model(model, model_path):
    # Dumping the model to file
    print ('Dumping model to ' + model_path)
    f = file(model_path, 'wb')

    # We dump only the parameters
    cPickle.dump(model.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)


def read_saved_model(model_path):
    saved_state = None
    if os.path.isfile(model_path):
        saved_state = cPickle.load(file(model_path, 'rb'))

    if saved_state is not None:
        # TODO We don't want saved state now.

        decision = raw_input('Read model from file? (y/N)')
        if decision != 'y':
            saved_state = None
    return saved_state


def detect_nan(i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break


########### Plotting #########################

def simple_plot_to_file(y, filename, ylim=None):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(y)
    if ylim is not None:
        axes.set_ylim(ylim)
    plt.savefig(filename)


def simple_plot(y, ylim=None):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(y)
    if ylim is not None:
        axes.set_ylim(ylim)
    plt.show()


############ Other ############################

class LazyWrapper(object):
    def __init__(self, func):
        self.func = func
    def __call__(self):
        try:
            return self.value
        except AttributeError:
            self.value = self.func()
            return self.value


def as_minibatch(t):
    return t.reshape(1, *t.shape)


def as_minibatch2(t):
    return as_minibatch(as_minibatch(t))


def flatten_row_major(arr):
    return arr.flatten(order='C')


def flatten_column_major(arr):
    return arr.flatten(order='F')


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#############################################
# TODO: Add numpy_rng here as option
def shared_normal(size, scale, name, rng=None, dtype=theano.config.floatX):
    if rng is None:
        value = np.random.normal(scale=scale, size=size)
    else:
        value = rng.normal(scale=scale, size=size)

    return theano.shared(value.astype(dtype), name=name)


def shared_uniform(size, scale, name, rng=None, dtype=theano.config.floatX):
    if rng is None:
        value = np.random.uniform(low=-scale, high=scale, size=size)
    else:
        value = rng.uniform(low=-scale, high=scale, size=size)

    return theano.shared(value.astype(dtype), name=name)


def shared_zeros(shape, name, dtype=theano.config.floatX):
    return theano.shared(np.zeros(shape, dtype=dtype), name=name)


def shared_ones(shape, name, dtype=theano.config.floatX):
    return theano.shared(np.ones(shape, dtype=dtype), name=name)
#############################################

def dict_desc(d):
    res = ''
    for key, val in d.iteritems():
        if len(res):
            res += ', '
        res += str(key) + ' = ' + str(val)
    return res


def list_desc(l):
    res = ''
    for key, val in l:
        if len(res):
            res += ', '
        res += str(key) + ' = ' + str(val)
    return res


TERMIOS = termios
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~TERMIOS.ICANON & ~TERMIOS.ECHO
    new[6][TERMIOS.VMIN] = 1
    new[6][TERMIOS.VTIME] = 0
    termios.tcsetattr(fd, TERMIOS.TCSANOW, new)
    c = None
    try:
        c = os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, TERMIOS.TCSAFLUSH, old)
    return c
################################################################


def set_numpy_print_opts():
    np.set_printoptions(threshold=300000)
    np.set_printoptions(precision=10)
    np.set_printoptions(suppress=True)


def draw_roc_curve(y_true, y_pred, filepath='out.svg'):
    print y_true, y_pred, y_true * y_pred
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.savefig(filepath)

################################################################


class ObjectAttributeReference(object):
    def __init__(self, obj, field_name):
        self.obj = obj
        self.field_name = field_name

    def get_value(self):
        return getattr(self.obj, self.field_name)

    def set_value(self, v):
        setattr(self.obj, self.field_name, v)


def ncycle(iterable, n):
    for item in itertools.cycle(iterable):
        for i in range(n):
            yield item


def merge_two_bunches(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return Bunch(z)


def unique(l):
    return list(set(l))

################################################################
def describe_device(dev):
    print dev.name(), dev.id, dev.pci_bus_id(), dev.compute_capability(), dev.total_memory()


def memory_usage():
    # Taken from http://stackoverflow.com/questions/897941/python-equivalent-of-phps-memory-get-usage
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result




