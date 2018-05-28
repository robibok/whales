import os
import pickle
import cPickle
from bunch import Bunch
from ml_utils import mkdir_p
import ml_utils


class Saver(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.objs_dir = os.path.join(self.save_dir, 'objs')
        self.init_dirs([self.objs_dir])

    def init_dirs(self, dirs):
        for dir in dirs:
            mkdir_p(dir)

    def get_path(self, subpath, name):
        dir = os.path.join(self.save_dir, subpath)
        mkdir_p(dir)
        filepath = os.path.join(dir, name)
        return filepath

    def save_obj(self, obj, name):
        print 'saving obj, ', name
        path = os.path.join(self.objs_dir, name)
        ml_utils.save_obj(obj, path)
        return path

    def load_obj(self, name):
        print 'loading obj, ', name
        path = os.path.join(self.objs_dir, name)
        return ml_utils.load_obj(path)

    def get_latest(self, dir_):
        raise RuntimeError()

    def save_sth(self, sth, subpath, name):
        if subpath is not None:
            subpath = os.path.join(self.save_dir, subpath)
            mkdir_p(subpath)
            filepath = os.path.join(subpath, name)
        else:
            filepath = os.path.join(self.save_dir, name)

        print '..saving to: ', filepath
        cPickle.dump(sth, file(filepath, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        print '..done'
        return filepath

    def open_file(self, subpath, name):
        if subpath is not None:
            subpath = os.path.join(self.save_dir, subpath)
            mkdir_p(subpath)

            filepath = os.path.join(subpath, name)
        else:
            filepath = os.path.join(self.save_dir, name)
        return Bunch(file=open(filepath, mode='w'), filepath=filepath)

    @classmethod
    def load_path(cls, path):
        print '..load path:', path
        return cPickle.load(file(path, 'rb'))


class ExperimentSaver(Saver):

    def __init__(self, save_dir):
        super(ExperimentSaver, self).__init__(save_dir)

        self.final_dir = os.path.join(self.save_dir, 'final')
        self.filters_dir = os.path.join(self.save_dir, 'filters')
        self.training_dir = os.path.join(self.save_dir, 'training')
        dirs = [self.final_dir, self.filters_dir, self.training_dir, self.objs_dir]

        self.init_dirs(dirs)

    def save_state(self, model, path):
        model.save_state(path)

    def save_state_new(self, model, path):
        model.save_state_new(path)

    def save_train_state(self, model, name):
        filepath = os.path.join(self.training_dir, name + '.3c')
        self.save_state(model, filepath)

    def save_train_state_new(self, model, name):
        filepath = os.path.join(self.training_dir, name + '.3c')
        self.save_state_new(model, filepath)

        return filepath

    def save_final_state(self, model, name):
        model.all_param_info()
        filepath = os.path.join(self.final_dir, name + '.3c')
        self.save_state(model, filepath)
        return filepath
