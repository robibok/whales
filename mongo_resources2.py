import copy
from bson import Binary
import cPickle
from enum import Enum


class ExperimentStatus(Enum):
        CREATED = 0
        QUEUED = 1
        RUNNING = 2
        COMPLETED = 3

class EpochData(object):
    def __init__(self, **kwargs):
        self.d = {}
        for key, value in kwargs.iteritems():
            self.set_field(key, value)

    def as_dict(self):
        return copy.copy(self.d)

    def set_field(self, field, value):
        self.d[field] = value

    def encode(self):
        return Binary(cPickle.dumps(self, protocol=2), 128)

    @classmethod
    def decode(cls, binary):
        return cPickle.loads(binary)

