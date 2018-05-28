from Queue import Empty
from collections import defaultdict
import os
import random
import time
import multiprocessing as mp
from multiprocessing import Process
from setproctitle import setproctitle
import traceback

from bunch import Bunch
import numpy as np

from ml_utils import chunks
import ml_utils


def get_dirs(root_dir):
    return [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

def get_files(root_dir):
    return [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]


def get_filepaths_subfolders(root_dir):
    subfolders = get_dirs(root_dir)
    all = []

    for subfolder in subfolders:
        class_dir = os.path.join(root_dir, subfolder)
        files = get_files(class_dir)

        for filename in files:
            all.append(Bunch(path=os.path.join(class_dir, filename),
                             strclass=subfolder,
                             filename=filename))
    return all


def get_filepaths_one_dir(root_dir):
    files = get_files(root_dir)
    all = []
    for filename in files:
        all.append(Bunch(path=os.path.join(root_dir, filename),
                         filename=filename))
    return all


def get_class_mapping(train_dir):
    examples = get_filepaths_subfolders(train_dir)

    strclasses = list(set(map(lambda a: a.strclass, examples)))
    # print strclasses
    strclasses = sorted(strclasses)
    class_mapping = {}
    for idx, strclass in enumerate(strclasses):
        class_mapping[strclass] = idx
    return class_mapping


def get_preexamples_test(root_dir):
    return get_filepaths_one_dir(root_dir)


def get_preexamples_subfolders(root_dir):
    return get_filepaths_subfolders(root_dir)

def get_random_idx_by_dist(dist):
    x = random.random()

    for idx, v in enumerate(dist):
        if x <= v:
            return idx
        x -= v
    return len(dist) - 1



def get_examples(preexamples, mode='all', set_class_idx=True, epoch_size=None, limit=None, dist=None, shuffle=True):
    assert(mode in ['all', 'dist'])

    name_to_ex = {}
    for ex in preexamples:
        name_to_ex[ex.filename] = ex

    if set_class_idx:
        strclasses = list(set(map(lambda a: a.strclass, preexamples)))
        by_strclass = defaultdict(list)

        for ex in preexamples:
            by_strclass[ex.strclass].append(ex)

        # print strclasses
        strclasses = sorted(strclasses)
        strclass_to_class_idx = {}
        class_idx_to_strclass = []
        for idx, strclass in enumerate(strclasses):
            strclass_to_class_idx[strclass] = idx
            class_idx_to_strclass.append(strclass)

    chosen = []

    if mode == 'dist':
        for i in xrange(epoch_size):
            class_idx = get_random_idx_by_dist(dist)

            strclass = class_idx_to_strclass[class_idx]
            ex = random.choice(by_strclass[strclass])
            other_ex = name_to_ex[get_other_name(ex.filename)]
            chosen.append(Bunch(path=ex.path, class_idx=class_idx, name=ex.filename, other_info=Bunch(other_ex=other_ex)))
    elif mode == 'all':
        for preexample in preexamples:
            other_ex = name_to_ex[get_other_name(preexample.filename)]
            ex = Bunch(path=preexample.path, name=preexample.filename, other_info=Bunch(other_ex=other_ex))

            if set_class_idx:
                ex.class_idx = strclass_to_class_idx[preexample.strclass]
            chosen.append(ex)

        if shuffle:
            random.shuffle(chosen)

    else:
        raise RuntimeError()

    if limit:
            chosen = chosen[:limit]

    return chosen


# This has to be picklable
class EndMarker(object):
    pass


class ExceptionMarker(object):
    def __init__(self, traceback):
        self.traceback = traceback

    def get_traceback(self):
        return self.traceback



class BufferedProcessor(object):
    def __init__(self, chunk_loader, buffer_size, add_to_queue_func, name):
        self.chunk_loader = chunk_loader
        self.buffer_size = buffer_size
        self.add_to_queue_func = add_to_queue_func
        self.name = name

    def get_iterator(self):
        def reader_process(chunk_loader, buffer, add_to_queue_func):
            # NOTICE:
            # We have to catch any exception raised in this process, and pass it to the parent
            # which is waiting on the Queue. Any better solution?

            try:
                setproctitle('cnn_buffered_processor' + self.name)
                idx = 0
                chunk_loader_iter = chunk_loader.get_iterator()

                while True:
                    try:
                        v = chunk_loader_iter.next()
                    except StopIteration:
                        break
                    add_to_queue_func(buffer, v)
                    idx += 1

                buffer.put(EndMarker())
            except Exception as e:
                buffer.put(ExceptionMarker(traceback.format_exc()))

        buffer = mp.Queue(maxsize=self.buffer_size)
        process = Process(target=reader_process, args=(self.chunk_loader, buffer, self.add_to_queue_func))
        process.start()
        TIMEOUT_IN_SECONDS = 600

        while True:
            #print 'BufferedProcessor', 'trying to get from the queue', buffer.qsize()
            try:
                v = buffer.get(timeout=TIMEOUT_IN_SECONDS)
            except Empty:
                print 'something is going wrong, could not get from buffer'
                raise

            if isinstance(v, EndMarker):
                break

            if isinstance(v, ExceptionMarker):
                raise RuntimeError(v.get_traceback())
            else:
                #print 'roz', buffer.qsize()
                yield v

        process.join()


class OutputDirector(object):
    def handle_result(self, res):
        raise RuntimeError()

    def handle_end(self):
        raise RuntimeError()

class MinibatchOutputDirector(object):
    def __init__(self, mb_size, data_shape, output_partial_batches=False):
        self.mb_size = mb_size
        self.output_partial_batches = output_partial_batches
        self.data_shape = data_shape

    def handle_begin(self):
        self._start_new_mb()

    def handle_result(self, res):
        self.current_batch.append(res)

        if 'x' in res:
            #print 'chu', self.current_mb_size, self.mb_size
            self.mb_x[self.current_mb_size] = res.x

        if 'y' in res:
            self.mb_y[self.current_mb_size] = res.y

        self.current_mb_size += 1

        if self.current_mb_size == self.mb_size:
            res = self._get_res()
            self._start_new_mb()
            return res
        else:
            return None

    def handle_end(self):
        print 'handle_end'
        if self.output_partial_batches:
            print 'OK', self.current_mb_size
            if len(self.current_batch) != 0:
                return self._get_res()
            else:
                return None
        else:
            print 'none'
            return None

    def _get_res(self):
        return Bunch(batch=self.current_batch,
                        mb_x=self.mb_x,
                        mb_y=self.mb_y)

    def _start_new_mb(self):
        self.current_mb_size = 0
        self.current_batch = []
        self.mb_x = np.zeros(shape=(self.mb_size,) + self.data_shape, dtype=ml_utils.floatX)
        self.mb_y = np.zeros(shape=(self.mb_size,), dtype=ml_utils.int32)

class MinibatchOutputDirector2(object):
    from ml_utils import floatX

    def __init__(self, mb_size, x_shape, y_shape, x_dtype=floatX, y_dtype=floatX, output_partial_batches=False):
        self.mb_size = mb_size
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.output_partial_batches = output_partial_batches

    def handle_begin(self):
        self._start_new_mb()

    def handle_result(self, res):
        self.current_batch.append(res)

        if 'x' in res:
            #print 'chu', self.current_mb_size, self.mb_size
            self.mb_x[self.current_mb_size] = res.x

        if 'y' in res:
            self.mb_y[self.current_mb_size] = res.y

        self.current_mb_size += 1

        if self.current_mb_size == self.mb_size:
            res = self._get_res()
            self._start_new_mb()
            return res
        else:
            return None

    def handle_end(self):
        print 'handle_end'
        if self.output_partial_batches:
            print 'OK', self.current_mb_size
            if len(self.current_batch) != 0:
                return self._get_res()
            else:
                return None
        else:
            print 'none'
            return None

    def _get_res(self):
        return Bunch(batch=self.current_batch,
                        mb_x=self.mb_x,
                        mb_y=self.mb_y)

    def _start_new_mb(self):
        self.current_mb_size = 0
        self.current_batch = []
        self.mb_x = np.zeros(shape=(self.mb_size,) + self.x_shape, dtype=self.x_dtype)
        self.mb_y = np.zeros(shape=(self.mb_size,) + self.y_shape, dtype=self.y_dtype)


class MultiprocessingChunkProcessor(object):
    def __init__(self, process_func, elements_to_process, output_director, chunk_size, pool_size=4, map_chunksize=4):
        """
        :param pre_func:
        :param process_func:
        :param post_func:
        :param examples:
        :param chunk_size: Yielded items will be of size 'chunk_size'.
        :param pool_size:
        :return:
        """
        self.process_func = process_func
        self.elements_to_process = elements_to_process
        self.output_director = output_director
        self.chunk_size = chunk_size
        self.pool_size = pool_size

    def get_iterator(self):
        pool = mp.Pool(self.pool_size)
        # pool = ThreadPool(pool_size)

        self.output_director.handle_begin()
        print 'Will try to pickle', type(self.process_func)
        for chunk in chunks(self.elements_to_process, self.chunk_size):
            chunk_results = pool.map(self.process_func, chunk, chunksize=4)
            for chunk_result in chunk_results:
                res = self.output_director.handle_result(chunk_result)
                if res is not None:
                    yield res

        print 100 * 'end'
        res = self.output_director.handle_end()
        if res is not None:
            yield res

        pool.close()
        pool.join()


class ProcessFunc(object):
    def __init__(self, process_func, *args, **kwargs):
        self.process_func = process_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, elem):
        setproctitle('cnn_worker_thread')
        recipe_result = self.process_func(elem, *self.args, **self.kwargs)
        return recipe_result


def get_iterator(process_func, elements_to_process, output_director, pool_size=4, buffer_size=20, chunk_size=100):

    def add_to_queue_func(buffer_queue, item):
        while buffer_queue.full():
            #print 'buffer is full, sleep for a while.', buffer_queue.qsize()
            time.sleep(5)
        #print 'put to buffer_queue'
        buffer_queue.put(item)


    return BufferedProcessor(
        MultiprocessingChunkProcessor(process_func, elements_to_process, output_director, chunk_size=chunk_size, pool_size=pool_size),
        buffer_size=buffer_size,
        add_to_queue_func=add_to_queue_func,
        name='get_valid_iterator').get_iterator()

def create_standard_iterator(process_func, elements_to_process, output_director, pool_size=4, buffer_size=20, chunk_size=100):

    def add_to_queue_func(buffer_queue, item):
        while buffer_queue.full():
            #print 'buffer is full, sleep for a while.', buffer_queue.qsize()
            time.sleep(5)
        #print 'put to buffer_queue'
        buffer_queue.put(item)


    return BufferedProcessor(
        MultiprocessingChunkProcessor(process_func, elements_to_process, output_director, chunk_size=chunk_size, pool_size=pool_size),
        buffer_size=buffer_size,
        add_to_queue_func=add_to_queue_func,
        name='get_valid_iterator').get_iterator()
