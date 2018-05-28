from __future__ import absolute_import
import cPickle

from bunch import Bunch
import numpy as np
import theano

from TheanoLib import utils


def key_diff(dict1, dict2):
    return [x for x in dict1.keys() if x not in dict2.keys()]

class SerializableModel(object):
    floatX = theano.config.floatX
    int32 = 'int32'

    def check_load_model(self):
        if self.args.load_model is not None:
            self.load_weights(self.args.load_model)

    def check_save_model(self):
         if self.args.save_model is not None:
            self.save_weights(self.args.save_model)

    def get_params(self):
        raise NotImplementedError()

    def all_param_info(self):
        params = self.get_params()
        weights = [param.get_value() for param in params]
        self.weight_info(params, weights)

    def save_state(self, filepath):
        print('Saving model to %s' % (filepath,))
        params = self.get_params()
        state_to_save = [param.get_value() for param in params]
        names = [param.name for param in params]
        print names
        cPickle.dump(state_to_save, file(filepath, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        print('done.')

    def load_state(self, filepath):
        print('Loading model from %s' % (filepath,))
        saved_state = cPickle.load(file(filepath, 'rb'))
        params = self.get_params()
        print 'len(params)', len(params), 'len(saved_state)', len(saved_state)

        assert(len(params) == len(saved_state))
        nof_params = 0
        for param, saved_param in zip(params, saved_state):
            print param.name, 'sum =', np.sum(saved_param), 'shape=', saved_param.shape
            nof_params += np.prod(saved_param.shape)
            param.set_value(saved_param)
        print 'Nof params in the mode =', nof_params

    def check_param_names(self):
        d = {}
        for param in self.get_params():
            if param.name in d:
                raise RuntimeError('Model should not have two params with the same name ' + param.name)
            d[param.name] += 1

    def save_state_new(self, filepath):
        print('Saving model to %s' % (filepath,))
        params = self.get_params()
        names = [param.name for param in params]
        print names
        state_to_save = [Bunch(value=param.get_value(), name=param.name) for param in params]
        cPickle.dump(state_to_save, file(filepath, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        print('done.')

    def load_state_new(self, filepath):
        print('Loading model from %s' % (filepath,))
        saved_state = cPickle.load(file(filepath, 'rb'))
        params = self.get_params()
        print 'len(params)', len(params), 'len(saved_state)', len(saved_state)
        saved_state_dict = {param.name: param for param in saved_state}

        params_dict = {param.name: param for param in params}

        print 'Params in current model: ', str(params_dict.keys())
        print 'Params in saved model: ', str(saved_state_dict.keys())

        print 'Params not present in saved model:', str(key_diff(params_dict, saved_state_dict))
        print 'Params present in saved model but not in current:', str(key_diff(saved_state_dict, params_dict))

        nof_params = 0
        for name, param in params_dict.iteritems():
            saved_value = None
            if name in saved_state_dict:
                saved_value = saved_state_dict[name].value

            sum = None if saved_value is None else np.sum(saved_value)
            print name, 'sum =', sum, 'shape=', param.get_value().shape
            nof_params += np.prod(param.get_value().shape)
            if saved_value is not None:
                param.set_value(saved_value)

        print 'Nof params in the mode =', nof_params



