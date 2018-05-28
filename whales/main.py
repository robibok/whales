import argparse
from collections import OrderedDict
from collections import defaultdict
import copy
import json
import os
import random
import sys

from bunch import Bunch
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from skimage.transform import AffineTransform
from sklearn.covariance import empirical_covariance
from tabulate import tabulate
from termcolor import colored
import theano
from theano.tensor.nnet import categorical_crossentropy
import theano.tensor as T

from TheanoLib import monitor_stats
import TheanoLib
from TheanoLib.utils import set_theano_fast_compile
from TheanoLib.utils import set_theano_fast_run
from image_utils import plot_image_to_file2, fetch_path_local, \
    resize_simple
from TheanoLib.modules import ActivationMonitoring
import ml_utils
from ExperimentManagement.mongo_resources2 import ExperimentStatus, EpochData
from saver import ExperimentSaver, Saver
from training_utils import  Context
from ml_utils import int32, repeat, Tee, unpack, npcast, init_seeds, epoch_header
from ml_utils import floatX, ArgparsePair
from TheanoLib import optimization
from ml_utils import elapsed_time_ms, elapsed_time_mins, elapsed_time_secs, start_timer, timestamp_str
from whales import transforms
from whales.architecture import construct_model
import dataloader2 as dataloader
import whales.dataloading

RED = np.asarray([0.99, 0, 0])
GREEN = np.asarray([0, 0.99, 0])
BLUE = np.asarray([0, 0, 0.99])

RED_int = np.asarray([254, 0, 0])
GREEN_int = np.asarray([0, 254, 0])
BLUE_int = np.asarray([0, 0, 254])
BLACK_int = np.asarray([0, 0, 0])

PRETTY1_int = np.asarray([245, 162, 111])
PRETTY1 = np.asarray([245 / 255.0, 162 / 255.0, 111 / 255.0])

PRETTY2_int = np.asarray([255, 220, 137])
PRETTY2 = np.asarray([255 / 255.0, 220 / 255.0, 137 / 255.0])


def print_fields_from_dict(d, fields):
    for field in fields:
        print field, ' = ', d[field]


def ArgparsePair(value):
    w = value.split('%')
    return int(w[0]), int(w[1])


from ExperimentManagement.dummy_trainer import DummyTrainer
class MyTrainer(DummyTrainer):

#from ExperimentManagement.trainer import Trainer
#class MyTrainer(Trainer):
    def create_functions(self, model, method=None, reg_params=None, batch_x=None, batch_y=None, what_stats=None,
                         to_compile=[], use_cpu=False, adv_alpha=None, adv_eps=None, make_some_noise=False, eta=0.01,
                         gamma=0.55, starting_time=0):
        res = {}

        x = T.tensor4('x', dtype=floatX)

        true_dist = T.matrix('true_dist', dtype=floatX)

        l2_reg_global = T.scalar(name='l2_reg', dtype=floatX)
        learning_rate = T.scalar(name='learning_rate', dtype=floatX)
        mb_size = T.scalar(name='mb_size', dtype=int32)

        def get_p(output, true_dist, reg_params):
            p_y_given_x = output
            print p_y_given_x.type.dtype, true_dist.type
            detailed_losses = {}
            for name, interval in zip(self.get_target_suffixes(self.TARGETS), self.get_intervals(self.TARGETS)):
                detailed_losses[name] = (categorical_crossentropy(p_y_given_x[:, interval[0]:interval[1]],
                                                                  true_dist[:, interval[0]:interval[1]]))

            loss = categorical_crossentropy(p_y_given_x, true_dist)
            print 'loss.type', loss.type
            mean_loss = T.mean(loss)

            print 'Reg params'
            for param, l2_reg_per_param in reg_params:
                print param.name, l2_reg_per_param

            l2_sum = sum(T.sum(param ** 2) * l2_reg_per_param for param, l2_reg_per_param in reg_params)
            l2_reg_cost = l2_sum * l2_reg_global
            cost = mean_loss + l2_reg_cost

            return Bunch(
                p_y_given_x=p_y_given_x,
                loss=loss,
                detailed_losses=detailed_losses,
                mean_loss=mean_loss,
                cost=cost,
                l2_reg_cost=l2_reg_cost
            )

        if ('train_function' in to_compile or
                    'train_with_monitor_function' in to_compile):
            resv_train = model.apply(Bunch(output=x), mode='train', use_cpu=use_cpu)
            train_p = get_p(resv_train.output, true_dist, reg_params)

            def to_new_opt_param(param):
                if isinstance(param, Bunch):
                    return param
                else:
                    return Bunch(param=param, lr=None)

            opt_params = map(to_new_opt_param, model.get_opt_params())

            # Normal training
            all_grads = T.grad(train_p.cost, map(lambda a: a.param, opt_params))

            updates, stats = TheanoLib.optimization.get_updates(all_parameters=opt_params,
                                                                all_grads=all_grads,
                                                                learning_rate=learning_rate,
                                                                momentum=0.9,
                                                                method=method,
                                                                get_diffs=True,
                                                                what_stats=what_stats,
                                                                add_noise=make_some_noise,
                                                                eta=eta,
                                                                gamma=gamma,
                                                                starting_time=starting_time
                                                                )
            train_outputs = {'loss': train_p.loss,
                             'detailed_losses': train_p.detailed_losses,
                             'mean_loss': train_p.mean_loss,
                             'cost': train_p.cost,
                             'p_y_given_x': train_p.p_y_given_x,
                             'l2_reg_cost': train_p.l2_reg_cost
                             }

            train_with_monitor_outputs = copy.copy(train_outputs)

            train_with_monitor_outputs['stats'] = stats

            if 'activation_monitoring' in resv_train:
                train_with_monitor_outputs['activation_monitoring'] = resv_train.activation_monitoring

            train_function = TheanoLib.function(
                inputs=[learning_rate, l2_reg_global, mb_size],
                outputs=train_outputs,
                updates=updates + resv_train.get('updates', []),
                givens={
                    x: batch_x[0:mb_size],
                    true_dist: batch_y[0:mb_size],
                },
                on_unused_input='warn',
            )

            train_with_monitor_function = TheanoLib.function(
                inputs=[learning_rate, l2_reg_global, mb_size],
                outputs=train_with_monitor_outputs,
                updates=updates + resv_train.get('updates', []),
                givens={
                    x: batch_x[0:mb_size],
                    true_dist: batch_y[0:mb_size],
                },
                on_unused_input='warn',
            )
            res['train_function'] = train_function
            res['train_with_monitor_function'] = train_with_monitor_function

        def post_apply_fun(module, nv, **kwargs):
            if 'module_outputs' not in nv:
                nv.module_outputs = OrderedDict()
            nv.module_outputs[module.name] = nv.output
            return nv

        if 'valid_function' in to_compile:

            resv_valid = model.apply(Bunch(output=x), mode='test', post_apply_fun=post_apply_fun, use_cpu=use_cpu)
            valid_p = get_p(resv_valid.output, true_dist, reg_params)

            valid_outputs = {'loss': valid_p.loss,
                             'mean_loss': valid_p.mean_loss,
                             'detailed_losses': valid_p.detailed_losses,
                             'cost': valid_p.cost,
                             'p_y_given_x': valid_p.p_y_given_x,
                             'l2_reg_cost': valid_p.l2_reg_cost
                             }

            if 'activation_monitoring' in resv_valid:
                valid_outputs['activation_monitoring'] = resv_valid.activation_monitoring

            valid_function = TheanoLib.function(
                inputs=[l2_reg_global, mb_size],
                outputs=valid_outputs,
                updates=resv_valid.get('updates', []),
                givens={
                    x: batch_x[0:mb_size],
                    true_dist: batch_y[0:mb_size],
                },
                on_unused_input='warn',
            )
            res['valid_function'] = valid_function

        if 'eval_function' in to_compile:
            resv_eval = model.apply(Bunch(output=x), mode='test', post_apply_fun=post_apply_fun, use_cpu=use_cpu)
            p_y_given_x = resv_eval.output

            eval_outputs = {'p_y_given_x': p_y_given_x, 'module_activations': resv_eval.module_outputs}
            eval_function = TheanoLib.function(
                inputs=[x],
                outputs=eval_outputs,
                updates=resv_eval.get('updates', []),
                givens={
                    mb_size: x.shape[0]
                },
                on_unused_input='warn',
            )
            res['eval_function'] = eval_function

        return res


class WhaleTrainer(MyTrainer):
    @classmethod
    def create_parser(cls):
        parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')

        parser.add_argument('--name', type=str, default=None, help='TODO')

        parser.add_argument('--load-arch-url', type=str, default=None, help='TODO')
        parser.add_argument('--load-params-url', type=str, default=None, help='TODO')
        parser.add_argument('--mode', type=str, default=None, help='TODO')

        # annos
        parser.add_argument('--slot-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--auto-slot-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--auto-indygo-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--ryj-conn-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--symetria-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--new-conn-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--widacryj-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--point1-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--point2-annotations-url', type=str, default=None, help='TODO')

        parser.add_argument('--target-name', type=str, default=None, help='TODO')

        # paths
        parser.add_argument('--train-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--test-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--train-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--test-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--mean-data-url', type=str, default=None, help='TODO')
        parser.add_argument('--pca-data-url', type=str, default=None, help='TODO')
        parser.add_argument('--global-saver-url', type=str, default='global', help='TODO')

        # dataloading
        parser.add_argument('--valid-pool-size', type=int, default=6, help='TODO')
        parser.add_argument('--train-pool-size', type=int, default=4, help='TODO')
        parser.add_argument('--test-pool-size', type=int, default=4, help='TODO')
        parser.add_argument('--train-buffer-size', type=int, default=100, help='TODO')
        parser.add_argument('--mb-size', type=int, default=1, help='TODO')
        parser.add_argument('--no-train-update', action='store_true', help='TODO')
        parser.add_argument('--n-epochs', type=int, default=1000000, help='TODO')

        parser.add_argument('--equalize', action='store_true', help='TODO')
        parser.add_argument('--indygo-equalize', action='store_true', help='TODO')

        parser.add_argument('--use-cpu', action='store_true', help='TODO')
        parser.add_argument('--no-est', action='store_true', help='TODO')

        parser.add_argument('--gen-crop1-train', action='store_true', help='TODO')
        parser.add_argument('--gen-crop1-test', action='store_true', help='TODO')
        parser.add_argument('--gen-crop2-train', action='store_true', help='TODO')
        parser.add_argument('--gen-crop2-test', action='store_true', help='TODO')
        parser.add_argument('--gen-submit', action='store_true', help='TODO')
        parser.add_argument('--gen-submit-mod', type=ArgparsePair, default=(0, 1), help='TODO')

        parser.add_argument('--write-valid-preds-all', action='store_true', help='TODO')
        parser.add_argument('--gen-valid-preds', action='store_true', help='TODO')
        parser.add_argument('--margin', type=int, default=40, help='TODO')

        parser.add_argument('--buckets', type=int, default=60, help='TODO')

        parser.add_argument('--diag', action='store_true', help='TODO')
        parser.add_argument('--real-valid-shuffle', action='store_true', help='TODO')
        parser.add_argument('--real-test-shuffle', action='store_true', help='TODO')

        parser.add_argument('--gen-saliency-map-url', type=str, default=None, help='TODO')
        parser.add_argument('--gen-train-saliency-map-n-random', type=int, default=None, help='TODO')
        parser.add_argument('--gen-valid-saliency-map-n-random', type=int, default=None, help='TODO')

        parser.add_argument('--gen-valid-annotations', type=ArgparsePair, default=None, help='TODO')
        parser.add_argument('--gen-train-valid-annotations', type=ArgparsePair, default=None, help='TODO')
        parser.add_argument('--gen-test-annotations', type=ArgparsePair, default=None, help='TODO')

        parser.add_argument('--verbose-valid', action='store_true', help='TODO')

        parser.add_argument('--invalid-cache', action='store_true', help='TODO')

        parser.add_argument('--pca-scale', type=float, default=None, help='TODO')
        parser.add_argument('--train-part', type=float, default=0.9, help='TODO')

        parser.add_argument('--report-case-th', type=float, default=None, help='TODO')
        parser.add_argument('--ssh-reverse-host', type=str, default=None, help='TODO')

        parser.add_argument('--adv-alpha', type=float, default=None, help='TODO')
        parser.add_argument('--adv-eps', type=float, default=None, help='TODO')
        parser.add_argument('--show-images', type=int, default=10, help='TODO')

        parser.add_argument('--valid-partial-batches', action='store_true', help='TODO')

        parser.add_argument('--FREQ1', type=int, default=80, help='TODO')
        parser.add_argument('--SAVE_FREQ', type=int, default=10, help='TODO')

        parser.add_argument('--do-pca', type=int, default=1, help='TODO')
        parser.add_argument('--do-mean', type=int, default=1, help='TODO')
        parser.add_argument('--do-dump', type=int, default=1, help='TODO')

        parser.add_argument('--n-classes', type=int, default=2, help='TODO')
        parser.add_argument('--loss-freq', type=int, default=1, help='TODO')
        parser.add_argument('--monitor-freq', type=int, default=9999999, help='TODO')
        parser.add_argument('--crop-h', type=int, default=448, help='TODO')
        parser.add_argument('--crop-w', type=int, default=448, help='TODO')
        parser.add_argument('--channels', type=int, default=3, help='TODO')
        parser.add_argument('--nof-best-crops', type=int, default=1, help='TODO')
        parser.add_argument('--n-samples-valid', type=int, default=None, help='TODO')
        parser.add_argument('--n-samples-test', type=int, default=None, help='TODO')

        parser.add_argument('--l2-reg-global', type=float, default=1.0, help='TODO')
        parser.add_argument('--glr', type=float, default=None, help='TODO')
        parser.add_argument('--valid-freq', type=int, default=5, help='TODO')
        parser.add_argument('--glr-burnout', type=int, default=19999999, help='TODO')
        parser.add_argument('--glr-decay', type=float, default=1.0, help='TODO')

        parser.add_argument('--arch', type=str, default=None, help='TODO')
        parser.add_argument('--log-name', type=str, default='log.txt', help='TODO')
        parser.add_argument('--no-train', action='store_true', help='TODO')
        parser.add_argument('--debug', action='store_true', help='TODO')
        parser.add_argument('--seed', type=int, default=None, help='TODO')
        parser.add_argument('--valid-seed', type=int, default=None, help='TODO')
        parser.add_argument('--method', type=str, default='rmsprop', help='TODO')
        parser.add_argument('--aug-params', type=str, default=None, help='TODO')
        parser.add_argument('--process-recipe-name', type=str, default=None, help='TODO')

        # hyperparams
        parser.add_argument('--dropout', type=float, default=None, help='TODO')
        parser.add_argument('--fc-l2-reg', type=float, default=None, help='TODO')
        parser.add_argument('--conv-l2-reg', type=float, default=None, help='TODO')
        parser.add_argument('--n-fc', type=float, default=None, help='TODO')
        parser.add_argument('--n-first', type=float, default=None, help='TODO')
        parser.add_argument('--make-some-noise', action='store_true', help='TODO')
        parser.add_argument('--eta', type=float, default=0.01, help='TODO')
        parser.add_argument('--gamma', type=float, default=0.55, help='TODO')

        # unknown
        parser.add_argument('--starting-time', type=int, default=0, help='TODO')
        parser.add_argument('--dummy-run', action='store_true', help='TODO')

        return parser

    def initialize(self):
        np.random.seed(None)
        seed = np.random.randint(0, 1000000000)
        if self.args.seed is not None:
            seed = self.args.seed

        self.exp.set_seed(seed)

        print 'Seed', seed
        self.seed = seed
        init_seeds(seed)
        self.numpy_rng = np.random.RandomState(seed)

        if self.args.valid_seed is None:
            self.valid_seed = random.randint(0, 10000)
        else:
            self.valid_seed = self.args.valid_seed

        self.exp.set_valid_seed(self.valid_seed)

        self.global_saver = Saver(self.url_translator.url_to_path(self.args.global_saver_path))
        if self.args.debug:
            print colored('Running --debug, fast compile', 'red')
            set_theano_fast_compile()
        else:
            print colored('Running fast run', 'red')

            set_theano_fast_run()

        if self.exp_dir_path:
            self.saver = ExperimentSaver(self.exp_dir_path)
            a = self.saver.open_file(None, self.args.log_name)
            self.log_file, filepath = a.file, a.filepath
            self.tee_stdout = Tee(sys.stdout, self.log_file)
            self.tee_stderr = Tee(sys.stderr, self.log_file)

            sys.stdout = self.tee_stdout
            sys.stderr = self.tee_stderr

            self.exp.set_log_url(self.url_translator.path_to_url(filepath))

        ml_utils.DEBUG = self.args.debug

    def pca_it(self, spec, recipes, process_recipe):
        MB_SIZE = 10
        process_func = dataloader.ProcessFunc(process_recipe, spec)
        output_director = dataloader.MinibatchOutputDirector2(MB_SIZE,
                                                              x_shape=(spec['target_channels'], spec['target_h'],
                                                                       spec['target_w']),
                                                              y_shape=(self.Y_SHAPE,)
                                                              )

        iterator = dataloader.create_standard_iterator(process_func, recipes, output_director,
                                                       pool_size=6, buffer_size=40, chunk_size=MB_SIZE * 3)

        print 'computing eigenvalues ...'

        X = np.concatenate([batch['mb_x'][0, ...].reshape((3, -1)).T for batch in iterator])
        n = X.shape[0]
        limit = 125829120
        if n > limit:
            X = X[np.random.randint(n, size=limit), :]
        print X.shape
        cov = empirical_covariance(X)
        print cov
        evs, U = eigh(cov)
        print evs
        print U

        return evs, U

    def set_targets(self):
        MANY_TARGETS = {
            'final':
                [
                    ('class', 447),
                    ('new_conn', 2),
                ],
            'final_no_conn':
                [
                    ('class', 447),
                ],
            'crop1':
                [
                    ('indygo_point1_x', self.args.buckets),
                    ('indygo_point1_y', self.args.buckets),
                    ('indygo_point2_x', self.args.buckets),
                    ('indygo_point2_y', self.args.buckets),
                    ('slot_point1_x', self.args.buckets),
                    ('slot_point1_y', self.args.buckets),
                    ('slot_point2_x', self.args.buckets),
                    ('slot_point2_y', self.args.buckets),
                ],
            'crop2':
                [
                    ('class', 447),
                    ('conn', 2),
                    ('indygo_point1_x', self.args.buckets),
                    ('indygo_point1_y', self.args.buckets),
                    ('indygo_point2_x', self.args.buckets),
                    ('indygo_point2_y', self.args.buckets),
                ]
        }
        self.TARGETS = MANY_TARGETS[self.args.target_name]
        print self.get_intervals(self.TARGETS)
        self.Y_SHAPE = sum(self.get_n_outs(self.TARGETS))

    @classmethod
    def get_n_outs(cls, TARGETS):
        return map(lambda a: a[1], TARGETS)

    @classmethod
    def get_target_suffixes(cls, TARGETS):
        return map(lambda a: a[0], TARGETS)

    @classmethod
    def get_y_shape(cls, TARGETS):
        return sum(map(lambda a: a[1], TARGETS))

    @classmethod
    def get_interval(cls, name, TARGETS):
        sum = 0
        for (a, b) in TARGETS:
            if a == name:
                return (sum, sum + b)
            sum += b
        return None

    @classmethod
    def norm_name(cls, key):
        if key[-4:] == '.jpg':
            key = key[0:-4]
        return key

    @classmethod
    def get_intervals(cls, TARGETS):
        res = []
        s = 0
        for _, n_out in TARGETS:
            res.append((s, s + n_out))
            s += n_out
        return res

    def estimeate_mean_var(self, iterator):
        channels = self.args.channels
        mean = np.zeros((channels,), dtype=floatX)
        var = np.zeros((channels,), dtype=floatX)
        n_examples = 0
        h, w = None, None

        for mb_idx, item in enumerate(iterator):
            print 'MB_IDX', mb_idx

            mb_x = item['mb_x']
            h = mb_x.shape[2]
            w = mb_x.shape[3]
            for idx in xrange(mb_x.shape[0]):
                n_examples += 1

                for channel in xrange(channels):
                    mean[channel] += np.sum(mb_x[idx, channel, ...])
                    var[channel] += np.sum(mb_x[idx, channel, ...] ** 2)

        mean /= n_examples * h * w
        var /= n_examples * h * w
        return mean, var

    def get_mean_std(self, spec, recipes, test_recipes):
        spec = copy.copy(spec)

        if self.args.no_est:
            mean = [0] * self.args.channels
            std = [1] * self.args.channels
        else:
            if self.args.mean_data_path is None:

                h = ml_utils.my_portable_hash([spec, len(recipes)])
                name = 'mean_std_{}'.format(h)
                print 'mean_std filename', name
                res = self.global_saver.load_obj(name)

                if res is None or self.args.invalid_cache:
                    print '..recomputing mean, std'

                    MB_SIZE = 40
                    process_func = dataloader.ProcessFunc(self.process_recipe, spec)
                    output_director = dataloader.MinibatchOutputDirector2(MB_SIZE,
                                                                          x_shape=(
                                                                          spec['target_channels'], spec['target_h'],
                                                                          spec['target_w']),
                                                                          y_shape=(self.Y_SHAPE,))

                    iterator = dataloader.create_standard_iterator(process_func, recipes, output_director,
                                                                   pool_size=6, buffer_size=40, chunk_size=3 * MB_SIZE)

                    mean, _ = self.estimeate_mean_var(iterator)
                    spec.mean = mean
                    iterator = dataloader.create_standard_iterator(process_func, recipes, output_director,
                                                                   pool_size=6, buffer_size=40, chunk_size=3 * MB_SIZE)
                    mean2, std_kw = self.estimeate_mean_var(iterator)
                    std = np.sqrt(std_kw)
                    print 'mean2', mean2
                    spec.std = std
                    iterator = dataloader.create_standard_iterator(process_func, test_recipes, output_director,
                                                                   pool_size=6, buffer_size=40, chunk_size=3 * MB_SIZE)
                    res = self.estimeate_mean_var(iterator)
                    print res

                    mean_data_path = self.global_saver.save_obj((mean, std), name)
                    self.exp.set_mean_data_url(self.url_translator.path_to_url(mean_data_path))
                else:
                    print '..using cached mean, std'
                    mean, std = res[0], res[1]
            else:
                mean_data = ml_utils.load_obj(self.args.mean_data_path)
                if len(mean_data) == 2:
                    mean, std = mean_data[0], mean_data[1]
                elif len(mean_data) == 3:
                     # historical compability
                     mean = np.asarray(mean_data)
                     std = np.asarray([255.0, 255.0, 255.0])

                else:
                    raise RuntimeError()

        return mean, std

    def init_shared(self, max_mb_size):
        x_dim = (max_mb_size, self.args.channels, self.args.crop_h, self.args.crop_w)
        y_dim = (max_mb_size, self.Y_SHAPE)

        self.x_sh = theano.shared(np.zeros(x_dim, floatX), name='x_sh')
        self.y_sh = theano.shared(np.zeros(y_dim, floatX), name='y_sh')

    def go(self, exp, args, exp_dir_path):
        self.url_translator = self.get_url_translator()
        self.args = args
        self.exp = exp

        self.init_command_receiver(self.args.ssh_reverse_host)
        self.start_exit_handler_thread(self.command_receiver)

        print 'ARGS', args

        self.n_classes = self.args.n_classes
        self.exp.set_status(ExperimentStatus.RUNNING)
        self.exp_dir_path = exp_dir_path

        print ' '.join(sys.argv)
        print args

        exp.set_name(self.args.name)
        self.create_bokeh_session()

        self.initialize()
        self.set_targets()
        self.ts = self.create_timeseries_and_figures()
        self.init_shared(self.args.mb_size)
        self.init_model()

        self.process_recipe = getattr(whales.dataloading, self.args.process_recipe_name)

        self.do_training()
        return 0

    def create_timeseries_and_figures(self, optim_state={}):
        FREQ1 = self.args.FREQ1

        channels = [
            # train channels
            'train_cost',
            'train_loss',
            'train_slot_loss',
            'train_indygo_loss',
            'train_l2_reg_cost',
            'train_per_example_proc_time_ms',
            'train_per_example_load_time_ms',
            'train_per_example_rest_time_ms',

            # valid channels
            'valid_loss',
            'valid_slot_loss',
            'valid_indygo_loss',

            'l2_reg_global',
            'train_epoch_time_minutes',
            'valid_epoch_time_minutes',
            'act_glr'
        ]

        for suff, _ in self.TARGETS:
            channels.append('train_loss_' + suff)
            channels.append('val_loss_' + suff)

        figures_schema = OrderedDict([

            # (channel_name, name_on_plot, frequency_of_updates)
            ('train', [
                ('train_cost', 'cost', FREQ1),
                ('train_loss', 'loss', FREQ1),
                ('train_slot_loss', 'slot_loss', FREQ1),
                ('train_indygo_loss', 'indygo_loss', FREQ1),
                ('train_l2_reg_cost', 'l2_reg_cost', FREQ1)] +
             [('train_loss_' + suff, 'loss_' + suff, FREQ1) for suff, _ in self.TARGETS]
             ),

            ('valid', [
                ('valid_loss', 'loss', 1),
                ('valid_slot_loss', 'slot_loss', 1),
                ('valid_indygo_loss', 'indygo_loss', 1)] +
             [('val_loss_' + suff, 'loss_' + suff, 1) for suff, _ in self.TARGETS]),

            ('train + valid', [
                ('train_loss', 'train_loss', FREQ1),
            ]),

            ('perf', [
                ('train_per_example_proc_time_ms', 'train_per_example_proc_ms', FREQ1),
                ('train_per_example_load_time_ms', 'train_per_example_load_ms', FREQ1),
                ('train_per_example_rest_time_ms', 'train_per_example_rest_ms', FREQ1)
            ]),

            ('perf_2', [
                ('train_epoch_time_minutes', 'train_epoch_time_minutes', 1),
                ('valid_epoch_time_minutes', 'valid_epoch_time_minutes', 1)
            ]),

            ('act_glr', [
                ('act_glr', 'act_glr', 1)
            ])

        ])

        return self._create_timeseries_and_figures(channels, figures_schema)

    def init_model(self):
        if self.args.load_arch_path is not None:
            print '..loading arch'
            self.model = self.saver.load_path(self.args.load_arch_path)
        else:
            params = {
                'channels': self.args.channels,
                'image_size': (self.args.crop_h, self.args.crop_w),
                'n_outs': self.get_n_outs(self.TARGETS),
                'conv_l2_reg': self.args.conv_l2_reg,
                'fc_l2_reg': self.args.fc_l2_reg,
                'dropout': self.args.dropout,
            }

            self.arch, self.model, self.reg_params = construct_model(self.args.arch, **params)

        print 'Saving arch'
        self.exp.set_nof_params(self.model.all_params_info().nof_params)
        self.exp.set_weights_desc(self.model.all_params_info().desc)

        if self.args.load_params_path is not None:
            print '..loading params', self.args.load_params_path
            self.model.load_state_new(self.args.load_params_path)

        params = self.model.get_params()
        filters = None
        for p in params:
            print p.name
            if p.name == 'MAIN.conv0.conv_filters':
                filters = p.get_value()
        path = self.saver.get_path('filters', 'filters.png')
        path_2 = self.saver.get_path('filters', 'filters_2.png')
        print type(filters)
        print 'path', path

    def create_iterator_factory(self):
        train_csv = pd.read_csv(self.args.train_csv_path, index_col='image')

        train_spec = Bunch({
            'equalize': self.args.equalize,
            'indygo_equalize': self.args.indygo_equalize,
            'target_h': self.args.crop_h,
            'target_w': self.args.crop_w,
            'target_channels': self.args.channels,
            'cropping_type': 'random',
            'mean': None,
            'std': None,
            'pca_data': None,
            'augmentation_params': getattr(transforms, self.args.aug_params),
            'margin': self.args.margin,
            'diag': self.args.diag,
            'buckets': self.args.buckets,
            'TARGETS': self.TARGETS
        })
        print list(train_csv.index)
        train_recipes, strclass_to_class_idx = unpack(self.create_recipes_old(train_csv, valid_seed=self.valid_seed,
                                                                              train_part=self.args.train_part),
                                                      'train_recipes', 'strclass_to_class_idx')
        print len(train_recipes)

        SAMPLING_SPLIT = 200

        if self.args.do_pca:
            if self.args.pca_data_path is None:
                print 'train_spec', train_spec
                h = ml_utils.my_portable_hash([train_spec, SAMPLING_SPLIT])

                name = 'pca_data_{}'.format(h)
                print 'pca_data filename', name
                pca_data = self.global_saver.load_obj(name)
                if pca_data is None or self.args.invalid_cache:
                    print '..recomputing pca_data'
                    pca_data = self.pca_it(train_spec, train_recipes[0:SAMPLING_SPLIT], self.process_recipe)
                    pca_data_path = self.global_saver.save_obj(pca_data, name)
                    self.exp.set_pca_data_url(self.url_translator.path_to_url(pca_data_path))

                else:
                    print '..using old pca_data'
            else:
                pca_data = ml_utils.load_obj(self.args.pca_data_path)
            print 'pca_data', pca_data

            train_spec['pca_data'] = pca_data
            train_spec['pca_scale'] = self.args.pca_scale


        if self.args.do_mean:
            mean, std = self.get_mean_std(train_spec, train_recipes[SAMPLING_SPLIT:2 * SAMPLING_SPLIT],
                                          train_recipes[2 * SAMPLING_SPLIT: 3 * SAMPLING_SPLIT], )
            self.mean = mean
            self.std = std
            print 'MEAN', mean, 'STD', std
            train_spec['mean'] = mean
            train_spec['std'] = std
        else:
            self.mean = np.zeros((3,), dtype=floatX)
            self.std = np.ones((3,), dtype=floatX)

        valid_spec = copy.copy(train_spec)

        test_spec = copy.copy(train_spec)
        process_recipe = self.process_recipe
        Y_SHAPE = self.Y_SHAPE

        class IteratorFactory(object):
            TEST_SPEC = test_spec
            TRAIN_SPEC = train_spec
            VALID_SPEC = valid_spec

            def get_strclass_to_class_idx(self):
                return strclass_to_class_idx

            def get_iterator(self, recipes, mb_size, spec, buffer_size=15, pool_size=5, chunk_mul=3,
                             output_partial_batches=False):
                print 'Create iterator!!!! pool_size = ', pool_size
                process_func = dataloader.ProcessFunc(process_recipe, spec)
                output_director = dataloader.MinibatchOutputDirector2(mb_size,
                                                                      x_shape=(
                                                                      spec['target_channels'], spec['target_h'],
                                                                      spec['target_w']),
                                                                      y_shape=(Y_SHAPE,),
                                                                      output_partial_batches=output_partial_batches)

                return dataloader.create_standard_iterator(
                    process_func,
                    recipes,
                    output_director,
                    pool_size=pool_size,
                    buffer_size=buffer_size,
                    chunk_size=chunk_mul * mb_size)

            def get_train_iterator(self, train_recipes, mb_size, pool_size=6, buffer_size=45):
                print 'Create train iterator!!!! pool_size = ', pool_size
                return self.get_iterator(train_recipes, mb_size, self.TRAIN_SPEC, pool_size=pool_size,
                                         buffer_size=buffer_size, chunk_mul=2)

            def get_valid_iterator(self, valid_recipes, mb_size, n_samples_valid, pool_size=6, buffer_size=30,
                                   real_valid_shuffle=False):
                valid_recipes_repeated = repeat(valid_recipes, n_samples_valid, mb_size)
                if real_valid_shuffle:
                    random.shuffle(valid_recipes_repeated)
                return self.get_iterator(valid_recipes_repeated, mb_size, self.VALID_SPEC, pool_size=pool_size,
                                         buffer_size=buffer_size)

            def get_test_iterator(self, test_recipes, mb_size, n_samples_test, buffer_size=5, real_test_shuffle=False, pool_size=5):
                test_recipes_repeated = repeat(test_recipes, n_samples_test, mb_size)
                if real_test_shuffle:
                    random.shuffle(test_recipes_repeated)
                print len(test_recipes_repeated)

                return self.get_iterator(test_recipes_repeated, mb_size, self.TEST_SPEC, buffer_size=buffer_size,
                                         output_partial_batches=True, pool_size=pool_size)

            def get_test_iterator2(self, test_recipes, mb_size, buffer_size=5, pool_size=5):
                return self.get_iterator(test_recipes, mb_size, self.TEST_SPEC, buffer_size=buffer_size,
                                         output_partial_batches=True, pool_size=pool_size)

        return IteratorFactory()

    def read_annotations(self):
        annotations = defaultdict(dict)

        def f_csv(pd_dataframe, anno_name):
            for key, d in pd_dataframe.iterrows():
                key = self.norm_name(self.norm_name(key))
                key = key + '.jpg'
                annotations[key][anno_name] = d['annotation']

        def f(l, anno_name):
            for el in l:
                if 'filename' in el:
                    key = el['filename']
                elif 'name' in el:
                    key = el['name']
                key = self.norm_name(self.norm_name(key))
                key = key + '.jpg'

                if 'annotation' in el:
                    annotations[key][anno_name] = el['annotation']

                if 'annotations' in el:
                    annotations[key][anno_name] = el['annotations']

        def f_crop(l, anno_name):
            x = 0
            for el in l:
                if 'filename' in el:
                    key = el['filename']
                elif 'name' in el:
                    key = el['name']
                key = self.norm_name(self.norm_name(key))
                key = key + '.jpg'

                if 'annotation' in el:
                    res = el['annotation']

                if 'annotations' in el:
                    res = el['annotations']

                res = list(sorted(res, key=lambda v: -v['score']))
                if self.args.nof_best_crops != -1:
                    res = res[0: self.args.nof_best_crops]

                annotations[key][anno_name] = res

        if self.args.slot_annotations_url is not None:
            slot_annotations_list = json.load(open(self.url_translator.url_to_path(self.args.slot_annotations_url), 'r'))
            f(slot_annotations_list, 'slot')

        if self.args.auto_indygo_annotations_url is not None:
            crop_annotations_list = json.load(
                open(self.url_translator.url_to_path(self.args.auto_indygo_annotations_url), 'r'))
            f_crop(crop_annotations_list, 'auto_indygo')

        if self.args.auto_slot_annotations_url is not None:
            crop_annotations_list = json.load(open(self.url_translator.url_to_path(self.args.auto_slot_annotations_url), 'r'))
            f_crop(crop_annotations_list, 'auto_slot')

        if self.args.ryj_conn_annotations_url is not None:
            ryj_conn_annotations_list = json.load(
                open(self.url_translator.url_to_path(self.args.ryj_conn_annotations_url), 'r'))
            f(ryj_conn_annotations_list, 'ryj_conn')

        if self.args.point1_annotations_url is not None:
            point1_annotations_list = json.load(open(self.url_translator.url_to_path(self.args.point1_annotations_url), 'r'))
            f(point1_annotations_list, 'point1')

        if self.args.point2_annotations_url is not None:
            point2_annotations_list = json.load(open(self.url_translator.url_to_path(self.args.point2_annotations_url), 'r'))
            f(point2_annotations_list, 'point2')

        if self.args.widacryj_csv_path is not None:
            widacryj_csv = pd.read_csv(self.args.widacryj_csv_path, index_col='name')
            f_csv(widacryj_csv, 'widacryj')

        if self.args.new_conn_csv_path is not None:
            new_conn_csv = pd.read_csv(self.args.new_conn_csv_path, index_col='name')
            f_csv(new_conn_csv, 'new_conn')

        if self.args.symetria_csv_path is not None:
            symetria_csv = pd.read_csv(self.args.symetria_csv_path, index_col='name')
            f_csv(symetria_csv, 'symetria')

        print 10 * 'ANNOTA'
        print len(annotations)

        return annotations

    def permute_preds(self, strclass_to_class_idx, preds):
        res = np.zeros((self.n_classes,), floatX)
        for level in xrange(self.n_classes):
            res[level] = preds[strclass_to_class_idx[level]]
        return res

    def write_preds(self, strclass_to_class_idx, preds, name, submit_file):
        assert (len(list(preds)) == 447)
        res = self.permute_preds(strclass_to_class_idx, preds)
        line = ','.join([name] + [str(a) for a in res])
        print >> submit_file, line

    def do_training(self):
        what_stats = [monitor_stats.ParamsAbsMean(),
                      monitor_stats.ParamsAbsMax(),
                      monitor_stats.GradsAbsMean(),
                      monitor_stats.GradsAbsMax(),
                      monitor_stats.DiffsL2(),
                      monitor_stats.DiffsAbsMax(),
                      monitor_stats.KrzynowekStat(),
                      monitor_stats.LpKrzynowekStat(),
                      ]
        self.annotations = self.read_annotations()

        to_compile = ['train_function', 'train_with_monitor_function', 'valid_function', 'eval_function']

        if self.args.no_train:
            to_compile.remove('train_function')
            to_compile.remove('train_with_monitor_function')
            to_compile.remove('valid_function')

        self.model_functions = self.create_functions(self.model, self.args.method, self.reg_params, self.x_sh,
                                                     self.y_sh, what_stats, to_compile=to_compile,
                                                     use_cpu=self.args.use_cpu, adv_alpha=self.args.adv_alpha,
                                                     adv_eps=self.args.adv_eps,
                                                     make_some_noise=self.args.make_some_noise,
                                                     eta=self.args.eta, gamma=self.args.gamma,
                                                     starting_time=self.args.starting_time)

        print 'Params info before training'
        print self.model.all_params_info()
        self.save_model(self.model, 'pre_training')

        iterator_factory = self.create_iterator_factory()
        strclass_to_class_idx = iterator_factory.get_strclass_to_class_idx()
        self.strclass_to_class_idx = strclass_to_class_idx

        act_params = Bunch()

        ct = Context()
        ct.trainer = self
        ct.args = self.args
        ct.act_params = act_params

        act_params.act_glr = self.args.glr

        if self.args.no_train is False:
            train_csv = pd.read_csv(self.args.train_csv_path, index_col='image')
            self.train_recipes, self.valid_recipes = unpack(
                self.create_recipes_old(train_csv, valid_seed=self.valid_seed, train_part=self.args.train_part),
                'train_recipes', 'valid_recipes')

            if self.args.show_images > 0:
                show_images_spec = copy.copy(iterator_factory.TRAIN_SPEC)
                show_images_iterator = iterator_factory.get_iterator(self.train_recipes, 10, show_images_spec)
                self.show_images(show_images_iterator, self.args.show_images, self.mean, self.std)

            train_function = self.model_functions['train_function']
            train_with_monitor_function = self.model_functions['train_with_monitor_function']
            valid_function = self.model_functions['valid_function']

            try:
                for epoch_idx in xrange(self.args.n_epochs):
                    random.shuffle(self.train_recipes)
                    random.shuffle(self.valid_recipes)

                    print epoch_header(epoch_idx)
                    epoch_train_recipes = self.train_recipes
                    random.shuffle(epoch_train_recipes)
                    epoch_valid_recipes = self.valid_recipes

                    epoch_params = {
                        'glr': act_params.act_glr,
                        'l2_reg_global': self.args.l2_reg_global,
                        'mb_size': self.args.mb_size
                    }

                    if epoch_idx >= self.args.glr_burnout:
                        act_params.act_glr *= self.args.glr_decay

                    n_train_batches = len(epoch_train_recipes) // epoch_params['mb_size']
                    n_valid_batches = len(epoch_valid_recipes) // epoch_params['mb_size']
                    train_iterator = iterator_factory.get_train_iterator(epoch_train_recipes, epoch_params['mb_size'],
                                                                         pool_size=self.args.train_pool_size,
                                                                         buffer_size=self.args.train_buffer_size)
                    idx = 0
                    timer = start_timer()

                    # We should delay creating the iterator because, it will start loading images
                    print 'valid_size', len(epoch_valid_recipes)

                    real_valid_len = len(epoch_valid_recipes)
                    for i in xrange(real_valid_len):
                        epoch_valid_recipes[i].idx = i

                    to_add = (epoch_params['mb_size'] - len(epoch_valid_recipes) % epoch_params['mb_size']) % \
                             epoch_params['mb_size']

                    if to_add and self.args.valid_partial_batches:
                        epoch_valid_recipes_fixed = epoch_valid_recipes + to_add * [epoch_valid_recipes[0]]
                    else:
                        epoch_valid_recipes_fixed = epoch_valid_recipes

                    print 'real_size', real_valid_len, len(epoch_valid_recipes_fixed)

                    valid_iterator = iterator_factory.get_valid_iterator(epoch_valid_recipes_fixed,
                                                                         epoch_params['mb_size'],
                                                                         self.args.n_samples_valid,
                                                                         pool_size=self.args.valid_pool_size,
                                                                         real_valid_shuffle=self.args.real_valid_shuffle)

                    # one_train_epoch
                    if self.args.no_train_update is False:
                        print 'batches', n_train_batches, n_valid_batches
                        train_losses, train_costs = unpack(
                            self.do_train_epoch(epoch_idx, epoch_params, train_function,
                                                train_with_monitor_function, train_iterator,
                                                self.model, what_stats, ct), 'train_losses', 'train_costs')
                    else:
                        train_losses = [-1.0]
                        train_costs = [-1.0]


                    # validation
                    if epoch_idx % self.args.valid_freq == 0 and epoch_idx > 0:
                        if self.args.real_valid_shuffle:
                            valid_losses = self.do_valid_epoch2(epoch_idx, epoch_params, valid_function, valid_iterator,
                                                                self.args.n_samples_valid,
                                                                real_valid_len=real_valid_len, ct=ct)
                        else:
                            valid_losses = self.do_valid_epoch(epoch_idx, epoch_params, valid_function, valid_iterator,
                                                               self.args.n_samples_valid, real_valid_len=real_valid_len,
                                                               ct=ct)
                    else:
                        valid_losses = [0]

                    # stats + saving
                    timestamp_str_ = timestamp_str()
                    file_name = 'epoch_' + str(epoch_idx)

                    if epoch_idx % self.args.SAVE_FREQ == 0:
                        model_path = self.save_model(self.model, file_name)
                    else:
                        model_path = None

                    epoch_data = EpochData(train_loss=np.mean(train_losses),
                                           valid_loss=np.mean(valid_losses),
                                           train_cost=np.mean(train_costs),
                                           train_costs=train_costs,
                                           train_losses=train_losses,
                                           valid_losses=valid_losses,
                                           epoch_params=epoch_params,
                                           model_path=self.url_translator.path_to_url(model_path))
                    self.exp.add_epoch_data(epoch_data.encode())

                    print 'Epoch', epoch_idx, 'train_losses', np.mean(train_losses), 'valid_losses', np.mean(
                        valid_losses)
                    print epoch_params


            except KeyboardInterrupt as e:
                print 'Early break.'

            print '..Training ended.'
        else:
            print '..No Training!!'

        if self.args.gen_crop1_test:
            part_idx, mod = self.args.gen_submit_mod
            test_names = ml_utils.get_part(self.create_test_names(self.args.test_csv_path), part_idx, mod)
            print 'train_name', test_names[:30]
            recipes = self.create_test_recipes(self.args.test_dir_path, test_names).test_recipes
            self.gen_crop1(ct, 'test_bbox.json', recipes, iterator_factory)

        if self.args.gen_crop1_train:
            part_idx, mod = self.args.gen_submit_mod
            train_names = ml_utils.get_part(self.create_train_names(self.args.train_csv_path), part_idx, mod)
            print 'train_name', train_names[:30]
            recipes = self.create_test_recipes(self.args.train_dir_path, train_names).test_recipes
            self.gen_crop1(ct, 'train_bbox.json', recipes, iterator_factory)

        if self.args.gen_crop2_test:
            part_idx, mod = self.args.gen_submit_mod
            test_names = ml_utils.get_part(self.create_test_names(self.args.test_csv_path), part_idx, mod)
            print 'train_name', test_names[:30]
            recipes = self.create_test_recipes(self.args.test_dir_path, test_names).test_recipes
            self.gen_crop2(ct, 'test_indygo.json', recipes, iterator_factory)

        if self.args.gen_crop2_train:
            part_idx, mod = self.args.gen_submit_mod
            train_names = ml_utils.get_part(self.create_train_names(self.args.train_csv_path), part_idx, mod)
            print 'train_name', train_names[:30]
            recipes = self.create_test_recipes(self.args.train_dir_path, train_names).test_recipes
            self.gen_crop2(ct, 'train_indygo.json', recipes, iterator_factory)

        if self.args.gen_submit:
            self.gen_submit(ct, iterator_factory, strclass_to_class_idx)

    def gen_submit(self, ct, iterator_factory, strclass_to_class_idx):
        part_idx, mod = self.args.gen_submit_mod
        eval_function = self.model_functions['eval_function']
        test_names = ml_utils.get_part(self.create_test_names(self.args.test_csv_path), part_idx, mod)
        print 'leeen', len(test_names)
        test_recipes = self.create_test_recipes(self.args.test_dir_path, test_names).test_recipes
        n = len(test_recipes)
        print 'test_recipes', len(test_recipes)
        for idx in xrange(len(test_recipes)):
            test_recipes[idx].idx = idx
        real_test_len = len(test_recipes)
        test_mb_size = self.args.mb_size
        print 'mb_size', test_mb_size, 'samples', self.args.n_samples_test

        ml_utils.REMOVE_ME()
        if self.args.real_test_shuffle:
            test_recipes = self.args.n_samples_test * test_recipes
            random.shuffle(test_recipes)

            test_iterator = iterator_factory.get_test_iterator2(
                test_recipes, test_mb_size, pool_size=self.args.test_pool_size)
            print 'get_test_iterator2', len(test_recipes)

        else:
            test_iterator = iterator_factory.get_test_iterator(
                test_recipes, test_mb_size, self.args.n_samples_test, pool_size=self.args.test_pool_size)
        submit_file1 = self.saver.open_file(None, 'submit1.csv').file
        submit_file2 = self.saver.open_file(None, 'submit2.csv').file
        f = open(self.args.test_csv_path, 'r')
        if part_idx == 0:
            submit_file1.write(f.readline())
            submit_file2.write(f.readline())
        done = 0
        print strclass_to_class_idx
        if self.args.real_test_shuffle:
            submit_iter = self.do_gen_submit2(eval_function, test_iterator, self.args.n_samples_test,
                                              real_test_len=real_test_len,
                                              mb_size=test_mb_size, ct=ct)
        else:
            submit_iter = self.do_gen_submit(eval_function, test_iterator, self.args.n_samples_test,
                                             test_mb_size, ct=ct)
        for name, (preds1, preds2) in submit_iter:
            self.write_preds(strclass_to_class_idx, preds1, name + '.jpg', submit_file1)
            print 'done', done
            done += 1
            if done == n:
                break


        submit_file1.close()
        submit_file2.close()

    def gen_crop1(self, ct, filename, recipes, iterator_factory):
        n = len(recipes)
        for idx in xrange(len(recipes)):
            recipes[idx].idx = idx

        recipes = self.fix_recipes_length(recipes, self.args.mb_size)
        iterator = iterator_factory.get_test_iterator(
            recipes, self.args.mb_size, self.args.n_samples_test, pool_size=self.args.test_pool_size)

        eval_function = self.model_functions['eval_function']
        test_mb_size = self.args.mb_size

        done = 0
        submit_iter = self.do_gen_submit_og(eval_function, iterator, self.args.n_samples_test,
                                            test_mb_size, ct=ct)

        def f(item):
            # W x n_samples
            recipe_res_list = item[0]
            preds = item[1]

            def get_bucket(preds, name):
                a = self.get_interval(name, self.TARGETS)
                return np.argmax(preds[a[0]: a[1]])

            all_wsps = defaultdict(list)
            for idx in xrange(preds.shape[1]):
                tform_res = recipe_res_list[idx].info['tform_res']
                tform_res_inv = AffineTransform(tform_res._inv_matrix)

                preds_curr = preds[:, idx]
                # NOTICE, 256 vs 224 here
                w = [
                    ('slot_point1_x', 'slot_point1_y', 'slot_point1'),
                    ('slot_point2_x', 'slot_point2_y', 'slot_point2'),
                ]
                for a, b, c in w:
                    wsp1 = whales.dataloading.rev_find_bucket(224, self.args.buckets, get_bucket(preds_curr, a))
                    wsp2 = whales.dataloading.rev_find_bucket(224, self.args.buckets, get_bucket(preds_curr, b))
                    wsp_orig = tform_res_inv((wsp1, wsp2))[0]
                    all_wsps[c].append(wsp_orig)

            return (recipe_res_list, all_wsps)

        ryj_preds = []

        for item in submit_iter:
            recipe_res_list, all_wsps = f(item)
            name = recipe_res_list[0].recipe.name

            path_orig = self.saver.get_path('crop1_imgs', name + '_orig.jpg')

            p1 = np.mean(np.asarray(all_wsps['slot_point1']), axis=0)
            p2 = np.mean(np.asarray(all_wsps['slot_point2']), axis=0)
            p1t = np.mean(np.asarray(all_wsps['slot_point1t']), axis=0)
            p2t = np.mean(np.asarray(all_wsps['slot_point2t']), axis=0)


            print p1, p2
            resp1 = p1
            resp2 = p2
            ryj_preds.append({'name': name,
                              'annotation': [
                                  {
                                      'score': 1,
                                      'coord1': [resp1[0], resp1[1]],
                                      'coord2': [resp2[0], resp2[1]]
                                  }
                              ]
                              })


            print 'done', done
            done += 1
            if done == n:
                break
            if self.args.dummy_run and done >= 30:
                break
        w = self.saver.open_file(None, filename)
        f = w.file
        print 'filepath', w.filepath
        json.dump(ryj_preds, f, indent=4, separators=(',', ': '))
        f.close()

    def fix_recipes_length(self, recipes, mb_size):
        to_add = (mb_size - len(recipes) % mb_size) % mb_size
        if to_add:
            recipes = recipes + recipes[:to_add]
        return recipes

    def gen_crop2(self, ct, filename, recipes, iterator_factory):
        n = len(recipes)
        for idx in xrange(len(recipes)):
            recipes[idx].idx = idx

        print 'len(recipes)', len(recipes)
        recipes = self.fix_recipes_length(recipes, self.args.mb_size)
        print 'len(recipes) after fix', len(recipes)
        iterator = iterator_factory.get_test_iterator(
            recipes, self.args.mb_size, self.args.n_samples_test, pool_size=self.args.test_pool_size)

        eval_function = self.model_functions['eval_function']

        test_mb_size = self.args.mb_size

        done = 0
        submit_iter = self.do_gen_submit_og(eval_function, iterator, self.args.n_samples_test,
                                            test_mb_size, ct=ct)

        indygo_preds = []

        def f(item):
            # W x n_samples
            recipe_res_list = item[0]
            preds = item[1]

            def get_bucket(preds, name):
                a = self.get_interval(name, self.TARGETS)
                return np.argmax(preds[a[0]: a[1]])

            all_wsps = defaultdict(list)
            for idx in xrange(preds.shape[1]):
                tform_res = recipe_res_list[idx].info['tform_res']
                tform_res_inv = AffineTransform(tform_res._inv_matrix)

                preds_curr = preds[:, idx]
                # NOTICE, 256 vs 224 here
                w = [('indygo_point1_x', 'indygo_point1_y', 'indygo_point1'),
                     ('indygo_point2_x', 'indygo_point2_y', 'indygo_point2'),
                     ]
                for a, b, c in w:
                    wsp1 = whales.dataloading.rev_find_bucket(256, self.args.buckets, get_bucket(preds_curr, a))
                    wsp2 = whales.dataloading.rev_find_bucket(256, self.args.buckets, get_bucket(preds_curr, b))
                    wsp_orig = tform_res_inv((wsp1, wsp2))[0]
                    all_wsps[c].append(wsp_orig)

            return (recipe_res_list, all_wsps)

        cnt = 0
        for item in submit_iter:
            recipe_res_list, all_wsps = f(item)

            name = recipe_res_list[0].recipe.name

            p1 = np.mean(np.asarray(all_wsps['indygo_point1']), axis=0)
            p2 = np.mean(np.asarray(all_wsps['indygo_point2']), axis=0)
            indygo_preds.append({'name': name,
                                 'annotation': [
                                     {
                                         'score': 1,
                                         'coord1': [p1[0], p1[1]],
                                         'coord2': [p2[0], p2[1]]
                                     }
                                 ]
                                 })

            if cnt < 30:
                orig_img = fetch_path_local(recipe_res_list[0].recipe.path)
                s = int(40 * (orig_img.shape[1] / 3000))
                path_orig = self.saver.get_path('crop2_imgs', name + '_orig.jpg')
                self.draw_point(orig_img, PRETTY1_int, p1, w=s)
                self.draw_point(orig_img, PRETTY2_int, p2, w=s)
                print 'show_images: saving to ', path_orig
                plot_image_to_file2(orig_img, path_orig)
                cnt += 1

            print 'done', done
            done += 1
            if done == n:
                break

        w = self.saver.open_file(None, filename)
        f = w.file
        print 'filepath', w.filepath
        json.dump(indygo_preds, f, indent=4, separators=(',', ': '))
        f.close()

    def draw_p(self, arr, color, coord1, coord2, w=2):
        def l(x): return x[0] + w, x[1] + w

        self.draw_rect(arr, color, coord1, l(coord1), w=2)
        self.draw_rect(arr, color, coord2, l(coord2), w=2)

    def draw_point(self, arr, color, coord, w=2):
        x1, y1 = coord
        arr[y1:y1 + w, x1:x1 + w] = color

    def draw_rect(self, arr, color, coord1, coord2, w=2):
        x1, y1 = coord1
        x2, y2 = coord2

        arr[y1:y2, x1: x1 + w] = color
        arr[y1:y2, x2: x2 + w] = color
        arr[y1: y1 + w, x1:x2] = color
        arr[y2: y2 + w, x1:x2] = color

    def do_gen_submit2(self, eval_function, test_iterator, n_samples, real_test_len, mb_size=32, ct=None):
        global_counter = np.zeros(shape=(real_test_len,), dtype='int32')
        p_y_given_x_global_all = np.zeros(shape=(real_test_len, self.Y_SHAPE, n_samples), dtype=floatX)
        names = {}
        c = 0

        for item in test_iterator:

            c += 1
            print c
            self.command_receiver.handle_commands(ct)

            results = item['batch']
            mb_x = item['mb_x']
            print 'mb_x', mb_x.shape, mb_x.dtype
            res = eval_function(mb_x)
            p_y_given_x = res['p_y_given_x']

            for j in xrange(len(results)):
                idx = results[j].recipe.idx
                names[idx] = results[j].recipe.name
                a = global_counter[idx]
                if a < n_samples:
                    p_y_given_x_global_all[results[j].recipe.idx, :, a] = p_y_given_x[j, :]
                    global_counter[idx] += 1
                    # break

        for j in xrange(real_test_len):
            samples = global_counter[j]
            print 'samples', samples
            p_y_given_x_merge = np.mean(p_y_given_x_global_all[j, :, :samples], axis=1)
            if j not in names:
                name = list(names.values())[0]
            else:
                name = names[j]
            yield (name, (p_y_given_x_merge[:447], p_y_given_x_merge[:447]))

    def do_gen_submit(self, eval_function, test_iterator, n_samples, mb_size=32, ct=None):
        try:
            while True:
                self.command_receiver.handle_commands(ct)

                p_y_given_x_ans = np.zeros(shape=(2, mb_size, self.Y_SHAPE), dtype=floatX)
                p_y_given_x_all = np.zeros(shape=(mb_size, self.Y_SHAPE, n_samples), dtype=floatX)

                item = None

                for sample_idx in xrange(n_samples):
                    item = test_iterator.next()
                    mb_x = item['mb_x']
                    print 'mb_x', mb_x.shape, mb_x.dtype
                    res = eval_function(mb_x)
                    p_y_given_x_all[:, :, sample_idx] = res['p_y_given_x'][:, inter[0]:inter[1]]
                    item = item

                p_y_given_x_ans[0, ...] = np.mean(p_y_given_x_all, axis=2)

                for idx in xrange(mb_size):
                    yield item.batch[idx].recipe.name, (p_y_given_x_ans[0, idx, :], p_y_given_x_ans[1, idx, :])

        except StopIteration:
            pass

        raise StopIteration

    def do_gen_submit_og(self, eval_function, test_iterator, n_samples, mb_size=32, ct=None):
        try:
            while True:
                p_y_given_x_all = np.zeros(shape=(mb_size, self.Y_SHAPE, n_samples), dtype=floatX)
                items = []
                for sample_idx in xrange(n_samples):
                    item = test_iterator.next()
                    items.append(item)
                    mb_x = item['mb_x']
                    print 'mb_x', mb_x.shape, mb_x.dtype
                    res = eval_function(mb_x)
                    p_y_given_x_all[:, :, sample_idx] = res['p_y_given_x'][:, :]

                for idx in xrange(mb_size):
                    yield (map(lambda item: item.batch[idx], items), p_y_given_x_all[idx, :, :])
        except StopIteration:
            pass

        raise StopIteration

    def do_valid_epoch2(self, epoch_idx, epoch_params, valid_function, valid_iterator, n_samples, real_valid_len, ct):
        print 'read_valid_l', real_valid_len
        valid_id = ml_utils.id_generator(5)
        valid_submit_file = self.saver.open_file(None,
                                                 'valid_submit_{epoch_idx}_{valid_id}.csv'.format(epoch_idx=epoch_idx,
                                                                                                  valid_id=valid_id)).file
        valid_all_samples_submit_file = self.saver.open_file(None,
                                                             'valid_all_samples_submit_{}.csv'.format(valid_id)).file

        valid_losses = defaultdict(list)
        valid_top5_acc = defaultdict(list)
        mb_idx = 0
        epoch_timer = start_timer()
        losses_list = []

        full_valid = []
        global_counter = np.zeros(shape=(real_valid_len,), dtype='int32')
        p_y_given_x_global_all = np.zeros(shape=(real_valid_len, self.Y_SHAPE, n_samples), dtype=floatX)
        global_correct_dist = np.zeros(shape=(real_valid_len, self.Y_SHAPE), dtype=floatX)

        sum_valid_len = 0
        for item in valid_iterator:
            self.command_receiver.handle_commands(ct)
            infos = []
            mb_xs = []
            mb_x, mb_y = item['mb_x'], item['mb_y']
            mb_xs.append(mb_x)
            results = item['batch']
            infos.append(map(lambda a: a.info, results))
            mb_y_corr = mb_y

            self.x_sh.set_value(mb_x)
            self.y_sh.set_value(mb_y)

            res = valid_function(epoch_params['l2_reg_global'], epoch_params['mb_size'])

            p_y_given_x = res['p_y_given_x']

            for j in xrange(len(results)):
                idx = results[j].recipe.idx
                a = global_counter[idx]
                if a < n_samples:
                    p_y_given_x_global_all[results[j].recipe.idx, :, a] = p_y_given_x[j, :]
                global_counter[idx] += 1
                global_correct_dist[idx, :] = mb_y_corr[j, :]


        ### END OF LOOP ###
        print 'real_valid_len', real_valid_len
        for j in xrange(real_valid_len):
            samples = global_counter[j]
            p_y_given_x_merge = np.mean(p_y_given_x_global_all[j, :, :samples], axis=1)
            correct_y = global_correct_dist[j]

            for suff, interval in zip(self.get_target_suffixes(self.TARGETS), self.get_intervals(self.TARGETS)):
                valid_loss = ml_utils.categorical_crossentropy(
                    p_y_given_x_merge[interval[0]:interval[1]],
                    correct_y[interval[0]:interval[1]])[0]

                if suff == 'class':
                    top5_accuracy = ml_utils.get_top_k_accuracy(
                        ml_utils.as_mb(p_y_given_x_merge[interval[0]:interval[1]]),
                        np.argmax(ml_utils.as_mb(correct_y[interval[0]:interval[1]]), axis=1), k=5)
                    print 'partial top5', top5_accuracy
                    valid_top5_acc[suff].append(top5_accuracy)

                if suff == 'class':
                    print np.mean(valid_loss)

                valid_losses[suff].append(valid_loss)

                if mb_idx % 10 == 0:
                    self.exp.update_ping()
                mb_idx += 1

        print valid_losses['class']
        for suff in self.get_target_suffixes(self.TARGETS):
            ts = getattr(self.ts, 'val_loss_' + suff)
            print 'suff', suff
            print 'LEEEEEEEEEN', len(valid_losses[suff])
            ts.add(np.mean(valid_losses[suff]))
            print 'top5 accuracy', np.mean(valid_top5_acc[suff])

        def g(l, c):
            vs = [np.mean(valid_losses[name]) for name in l]
            ts = getattr(self.ts, c)
            v = np.sum(vs)
            print 'add_to', c
            ts.add(v)
            return v

        if self.args.mode == 'crop1':
            slot_loss = g(['slot_point1_x', 'slot_point1_y', 'slot_point2_x', 'slot_point2_y'], 'valid_slot_loss')
            indygo_loss = g(['indygo_point1_x', 'indygo_point1_y', 'indygo_point2_x', 'indygo_point2_y'],
                            'valid_indygo_loss')
            valid_loss = 0.5 * slot_loss + 0.5 * indygo_loss
        elif self.args.mode == 'crop2':
            valid_loss = np.mean(valid_losses['class'])
        elif self.args.mode == 'final':
            valid_loss = np.mean(valid_losses['class'])
        else:
            raise RuntimeError()

        print 'setting valid loss', valid_loss
        self.ts.valid_loss.add(valid_loss)

        self.ts.valid_epoch_time_minutes.add(elapsed_time_mins(epoch_timer))
        valid_submit_file.close()
        valid_all_samples_submit_file.close()

        if self.args.write_valid_preds_all:
            self.saver.save_obj(full_valid, 'full_valid.3c')

        return valid_loss

    def do_valid_epoch(self, epoch_idx, epoch_params, valid_function, valid_iterator, n_samples, real_valid_len, ct):
        print 'read_valid_l', real_valid_len
        lines_done = 0
        lines_done_2 = 0
        valid_id = ml_utils.id_generator(5)
        valid_submit_file = self.saver.open_file(None,
                                                 'valid_submit_{epoch_idx}_{valid_id}.csv'.format(epoch_idx=epoch_idx,
                                                                                                  valid_id=valid_id)).file
        valid_all_samples_submit_file = self.saver.open_file(None,
                                                             'valid_all_samples_submit_{}.csv'.format(valid_id)).file

        valid_losses = defaultdict(list)
        valid_top5_acc = defaultdict(list)
        mb_idx = 0
        epoch_timer = start_timer()
        verbose_valid = self.args.verbose_valid

        losses = defaultdict(list)
        losses_list = []
        results = None

        full_valid = []
        try:
            p_y_given_x_ans = np.zeros(shape=(1, epoch_params['mb_size'], self.Y_SHAPE), dtype=floatX)
            p_y_given_x_all = np.zeros(shape=(epoch_params['mb_size'], self.Y_SHAPE, n_samples), dtype=floatX)
            vidx = 0
            while True:
                self.command_receiver.handle_commands(ct)
                mb_size = epoch_params['mb_size']
                recipes = None
                infos = []
                mb_xs = []
                for samples_idx in xrange(n_samples):
                    item = valid_iterator.next()
                    mb_x, mb_y = item['mb_x'], item['mb_y']
                    mb_xs.append(mb_x)
                    results = item['batch']
                    infos.append(map(lambda a: a.info, results))
                    current_mb_size = len(results)
                    mb_y_corr = mb_y

                    self.x_sh.set_value(mb_x)
                    self.y_sh.set_value(mb_y)

                    res = valid_function(epoch_params['l2_reg_global'], epoch_params['mb_size'])

                    p_y_given_x = res['p_y_given_x']

                    p_y_given_x_all[:, :, samples_idx] = p_y_given_x

                p_y_given_x_ans[0, ...] = np.mean(p_y_given_x_all, axis=2)

                for j in xrange(mb_x.shape[0]):
                    if lines_done < real_valid_len:
                        name = results[j].recipe.name

                        if self.args.write_valid_preds_all:
                            for sample_idx in xrange(n_samples):
                                preds = p_y_given_x_all[j, :, sample_idx]
                                full_valid.append(Bunch(
                                    sample_idx=sample_idx,
                                    name=name,
                                    info=infos[sample_idx][j],
                                    preds=self.permute_preds(self.strclass_to_class_idx, preds[:447])
                                ))

                                p = '{name}_{sample_idx}.jpg'.format(name=name, sample_idx=sample_idx)

                                path = self.saver.get_path('full_valid_imgs', p)
                                img = np.rollaxis(mb_xs[sample_idx][j, ...], axis=0, start=3)

                                img = self.rev_img(img, self.mean, self.std)

                                loss = -(mb_y_corr[j, :447] * np.log(preds[:447])).sum()
                                print 'show_images: saving to ', path, 'loss', loss
                                plot_image_to_file2(img, path)

                                self.write_preds(self.strclass_to_class_idx, preds[:447], name,
                                                 valid_all_samples_submit_file)

                            preds = p_y_given_x_ans[0, j, ...]
                            self.write_preds(self.strclass_to_class_idx, preds[:447], name, valid_submit_file)

                        lines_done += 1

                for suff, interval in zip(self.get_target_suffixes(self.TARGETS), self.get_intervals(self.TARGETS)):
                    temp_lines_done_2 = lines_done_2
                    valid_loss = ml_utils.categorical_crossentropy(p_y_given_x_ans[0, :, interval[0]:interval[1]],
                                                                   mb_y_corr[:, interval[0]:interval[1]])

                    if suff == 'class':
                        top5_accuracy = ml_utils.get_top_k_accuracy(p_y_given_x_ans[0, :, interval[0]:interval[1]],
                                                                    np.argmax(mb_y_corr[:, interval[0]:interval[1]],
                                                                              axis=1), k=5)
                        print 'partial top5', top5_accuracy
                        valid_top5_acc[suff].append(top5_accuracy)

                    if suff == 'class':
                        print np.mean(valid_loss)

                    for j in xrange(mb_x.shape[0]):
                        if temp_lines_done_2 < real_valid_len:
                            valid_losses[suff].append(valid_loss[j])
                            temp_lines_done_2 += 1
                        else:
                            break

                lines_done_2 = temp_lines_done_2

                if mb_idx % 10 == 0:
                    self.exp.update_ping()
                mb_idx += 1

        except StopIteration:
            pass

        print valid_losses['class']

        for suff in self.get_target_suffixes(self.TARGETS):
            ts = getattr(self.ts, 'val_loss_' + suff)
            print 'suff', suff
            print 'LEEEEEEEEEN', len(valid_losses[suff])
            ts.add(np.mean(valid_losses[suff]))
            print 'top5 accuracy', np.mean(valid_top5_acc[suff])

        losses_list = sorted(losses_list, key=lambda b: b.loss)
        for b in losses_list:
            print b.loss, b.recipe.name

        self.ts.valid_epoch_time_minutes.add(elapsed_time_mins(epoch_timer))
        valid_submit_file.close()
        valid_all_samples_submit_file.close()

        if self.args.write_valid_preds_all:
            self.saver.save_obj(full_valid, 'full_valid.3c')

        return valid_losses[self.get_target_suffixes(self.TARGETS)[0]]

    def rev_img(self, img, mean, std):
        img *= std
        img += mean

        img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        return img

    def show_images(self, train_iterator, n, mean, std):
        k = 0
        for mb_idx, item in enumerate(train_iterator):
            print 'item', type(item)
            mb_x, mb_y = item['mb_x'], item['mb_y']
            batch = item['batch']
            mb_size = len(batch)

            for j in xrange(mb_size):
                img = np.rollaxis(mb_x[j, ...], axis=0, start=3)
                name = batch[j].recipe.name

                path_suffix = ''
                class_inter = self.get_interval('class', self.TARGETS)
                if class_inter is not None:
                    class_idx = np.argmax(mb_y[j, class_inter[0]:class_inter[1]])
                    path_suffix += '_class_idx_' + str(class_idx)

                crop2_inter = self.get_interval('crop2', self.TARGETS)
                if crop2_inter is not None:
                    crop2_idx = np.argmax(mb_y[j, crop2_inter[0]:crop2_inter[1]])
                    path_suffix += '_crop2_idx_' + str(crop2_idx)

                conn_inter = self.get_interval('conn', self.TARGETS)
                if conn_inter is not None:
                    ryj_conn_idx = np.argmax(mb_y[j, conn_inter[0]: conn_inter[1]])
                    if mb_y[j][ryj_conn_idx + conn_inter[0]] < 0.5:
                        ryj_conn_idx = -1
                else:
                    ryj_conn_idx = -2

                filename = 'img_{mb_idx}_{name}'.format(mb_idx=mb_idx, name=name) + path_suffix + '.jpg'
                path = self.saver.get_path('imgs', filename)

                print 'show_images: saving to ', path
                plot_image_to_file2(img, path)
                k += 1
                if k >= n:
                    return

    def do_train_epoch(self, epoch_idx, epoch_params, train_function, train_with_monitor_function, train_iterator,
                       model,
                       what_stats, ct):
        reported_cases = 0

        self.ts.act_glr.add(epoch_params['glr'])
        mb_size = epoch_params['mb_size']
        epoch_timer = start_timer()

        train_losses, train_costs = [], []

        load_timer = start_timer()
        whole_batch_timer = start_timer()

        class_loss_sum = np.zeros((self.n_classes,), floatX)
        class_n = np.zeros((self.n_classes,), floatX)

        for mb_idx, item in enumerate(train_iterator):
            if self.args.dummy_run and mb_idx >= 4:
                print 'DummyAction'
                break
            self.command_receiver.handle_commands(ct)
            mb_x, mb_y = item['mb_x'], item['mb_y']

            self.x_sh.set_value(mb_x)
            self.y_sh.set_value(mb_y)

            load_time_per_example = float(elapsed_time_ms(load_timer)) / mb_size
            print 'load time per example', load_time_per_example
            self.ts.train_per_example_load_time_ms.add(load_time_per_example)

            timer = start_timer()

            if mb_idx % self.args.monitor_freq == 0 and mb_idx:
                call_timer = start_timer()
                res = train_with_monitor_function(epoch_params['glr'], epoch_params['l2_reg_global'],
                                                  epoch_params['mb_size'])
                call_time_per_example = float(elapsed_time_ms(call_timer)) / mb_size
                self.ts.train_per_example_proc_time_ms.add(call_time_per_example)

                print 'Monitor:'
                if 'stats' in res:
                    monitor_stats.analyze_results(model.get_opt_params(), res['stats'], what_stats)

                print 'Activation monitoring'
                activation_monitoring = res.get('activation_monitoring', [])
                print 'Buckets'
                buckets = ActivationMonitoring.get_buckets()
                headers = ['name', 'mean', 'stdev'] + map(lambda (a, b): str(a) + ',' + str(b), buckets)
                table = []
                for a in activation_monitoring:
                    row = [a['name'], a['mean'], a['stdev']]
                    row += map(lambda w: str(w), a['histogram'])
                    print np.sum(a['histogram'])
                    table.append(row)

                print tabulate(table, headers, tablefmt='simple')

            else:
                call_timer = start_timer()
                res = train_function(epoch_params['glr'], epoch_params['l2_reg_global'], epoch_params['mb_size'])
                call_time_per_example = float(elapsed_time_ms(call_timer)) / mb_size
                self.ts.train_per_example_proc_time_ms.add(call_time_per_example)
                print 'call time per example', call_time_per_example


            to_print = ['mean_loss', 'cost', 'l2_reg_cost']

            loss2 = np.zeros(shape=(mb_size, 1), dtype=floatX)
            loss2[:, 0] = res['loss']

            print_fields_from_dict(res, to_print)
            self.ts.train_cost.add(res['cost'])

            for suff in self.get_target_suffixes(self.TARGETS):
                loss = res['detailed_losses'][suff]
                ts = getattr(self.ts, 'train_loss_' + suff)
                ts.add(np.mean(loss))

            def g(l, c):
                vs = [np.mean(res['detailed_losses'][name]) for name in l]
                ts = getattr(self.ts, c)
                v = np.sum(vs)
                ts.add(v)
                return v

            if self.args.mode == 'crop1':
                slot_loss = g(['slot_point1_x', 'slot_point1_y', 'slot_point2_x', 'slot_point2_y'], 'train_slot_loss')
                indygo_loss = g(['indygo_point1_x', 'indygo_point1_y', 'indygo_point2_x', 'indygo_point2_y'],
                                'train_indygo_loss')
                train_loss = 0.5 * slot_loss + 0.5 * indygo_loss
            elif self.args.mode == 'crop2':
                train_loss = np.mean(res['detailed_losses']['class'])
            elif self.args.mode == 'final':
                train_loss = np.mean(res['detailed_losses']['class'])
            else:
                raise RuntimeError()

            self.ts.train_loss.add(train_loss)
            train_losses.append(train_loss)

            self.ts.train_l2_reg_cost.add(res['l2_reg_cost'])

            if mb_idx % self.args.loss_freq == 0 and mb_idx >= self.args.loss_freq:
                print 'mb_idx', mb_idx
                print epoch_params
                print 'Current batch:', 'cost', res['cost'], 'mean_loss', res['mean_loss'], 'l2_reg_cost', res[
                    'l2_reg_cost']
                print 'Avgs:', 'loss', np.mean(self.ts.train_loss.last_n(self.args.loss_freq)), 'cost', np.mean(
                    self.ts.train_cost.last_n(self.args.loss_freq))
                print 'Time elapsed since beginning of epoch:', elapsed_time_secs(epoch_timer)

            if mb_idx % 10 == 0:
                self.exp.update_ping()

            load_timer = start_timer()
            one_add_timer = start_timer()

            rest_time_per_example = float(
                elapsed_time_ms(whole_batch_timer)) / mb_size - call_time_per_example - load_time_per_example
            self.ts.train_per_example_rest_time_ms.add(rest_time_per_example)
            whole_batch_timer = start_timer()

        self.ts.train_epoch_time_minutes.add(elapsed_time_mins(epoch_timer))

        return Bunch(train_losses=train_losses,
                     train_costs=train_costs)

    def create_train_names(self, train_csv_path):
        train_csv = pd.read_csv(train_csv_path, index_col='image')
        train_names = []
        for name in list(train_csv.index):
            train_names.append(name)
        return train_names

    def create_test_names(self, test_csv_path):
        test_csv = pd.read_csv(test_csv_path, index_col='Image')
        test_names = map(lambda name: self.norm_name(name), list(test_csv.index))
        return test_names

    def get_annotations(self, name):
        if self.annotations is not None:
            annotations = self.annotations[name + '.jpg']
        else:
            annotations = None

        return annotations

    def create_test_recipes(self, base_dir, test_names):
        recipes = []
        for name in test_names:
            print name
            name = self.norm_name(name)
            annotations = self.get_annotations(self.norm_name(name))
            recipe = Bunch(name=name, path=os.path.join(base_dir, name + '.jpg'),
                           annotations=annotations, fetch_true_dist=False)
            recipes.append(recipe)

        return Bunch(test_recipes=recipes)

    def create_recipes_old(self, train_csv, valid_seed, train_part=0.9):
        rng = random.Random()
        rng.seed(valid_seed)

        recipes = []

        whale_ids = list(train_csv['level'].unique())
        count = defaultdict(int)
        strclass_to_class_idx = {}
        class_idx_to_strclass = {}
        strclasses = whale_ids

        for idx, strclass in enumerate(strclasses):
            strclass_to_class_idx[strclass] = idx
            class_idx_to_strclass[idx] = strclass

        for name in list(train_csv.index):
            whale_id = train_csv.loc[name, 'level']
            count[strclass_to_class_idx[whale_id]] += 1

        for name in list(train_csv.index):
            annotations = self.get_annotations(name)

            whale_id = train_csv.loc[name, 'level']
            recipe = Bunch(name=name,
                           path=os.path.join(self.args.train_dir_path, name + '.jpg'),
                           annotations=annotations,
                           class_idx=strclass_to_class_idx[whale_id],
                           fetch_true_dist=True
                           )

            recipes.append(recipe)

        rng.shuffle(recipes)
        n = len(recipes)
        split_point = int(train_part * n)
        train_recipes = recipes[:split_point]
        valid_recipes = recipes[split_point:]

        print ','.join(map(lambda a: a.name, valid_recipes))

        print 'VALID EXAMPLES'
        if len(valid_recipes) > 10:
            print valid_recipes[:10]

        return Bunch(train_recipes=train_recipes,
                     valid_recipes=valid_recipes,
                     strclass_to_class_idx=strclass_to_class_idx)


def my_fetch(recipe, spec):
    img_orig = fetch_path_local(recipe.path)
    ((x1, y1), (x2, y2)) = recipe.coords
    img_crop = img_orig[y1:y2, x1:x2]
    img_ready = resize_simple(img_crop, spec.target_w_h)
    img_ready = (npcast(img_ready, floatX) - spec.mean) / spec.std
    img_ready = np.rollaxis(img_ready, 2)
    return Bunch(x=img_ready, coords=recipe.coords)


if __name__ == '__main__':
    trainer = WhaleTrainer()
    paths_to_dump = ['whales']
    pythonpaths = ['']
    default_owner = 'kaggler'

    sys.exit(trainer.main(default_owner=default_owner,
                          paths_to_dump=paths_to_dump,
                          pythonpaths=pythonpaths))
