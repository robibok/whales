import sys

from bunch import Bunch
import numpy as np

from TheanoLib.init import Uniform, Normal
from TheanoLib.modules import Sequential, CudaConv2dCudnn, ConvBatchNormalization, ApplyActivation, ActivationMonitoring, \
    MaxPoolingCudnn, Flatten, Dropout, Dense, rectify, identity, Softmax, relu


# Hacks
########################################################3

acth, actw, actchannels, llayer = [28], [28], [1], [None]


def init(channels, h, w):
    global acth
    global actw
    global actchannels
    global llayer
    acth, actw, actchannels, llayer = [h], [w], [channels], [None]


def add_layer(seq, layer):
    seq.add(layer)
    try:
        print 'tatttt'
        acth[0] = layer.get_output_height()
        actw[0] = layer.get_output_width()
        actchannels[0] = layer.get_output_filters()
        print acth[0], actw[0]
    except Exception as e:
        pass


def add_layer2(node):
    try:
        llayer[0] = node
        acth[0] = node.layer.get_output_height()
        actw[0] = node.layer.get_output_width()
        actchannels[0] = node.layer.get_output_filters()
    except Exception as e:
        pass


def act_h():
    return acth[0]


def act_w():
    return actw[0]


def act_channels():
    return actchannels[0]


def last_layer():
    return llayer[0]


######################################################


# def construct_model(arch_name, **kwargs):
#     mod = sys.modules[__name__]
#     return getattr(mod, arch_name)(**kwargs)


def construct_model(arch_name, **kwargs):
    module_name = 'whales.architectures.' + arch_name
    mod = __import__(module_name, fromlist=[''])
    return getattr(mod, 'create')(**kwargs)


def w(n):
        return 1.0 / np.sqrt(n)

def create_conv_column_rob(cfg, prefix=''):
    net = Sequential(name=prefix + 'sequential')
    counter = {'CONV': 0, 'CBN': 0, 'REC': 0, 'MON': 0, 'MAXP': 0}
    print 'len', len(cfg)
    for layer_cfg in cfg:
        type = layer_cfg.type
        print 'create', type

        counter[type] += 1
        name = prefix + type + str(counter[type])

        if type == 'CONV':
            add_layer(net, CudaConv2dCudnn(act_h(), act_w(),
                                                workmem='small',
                                                n_input_channels=act_channels(),
                                                n_filters=layer_cfg.n_filters,
                                                kernel_size=(layer_cfg.kernel_size),
                                                kernel_stride=(layer_cfg.kernel_stride),
                                                padding=(layer_cfg.padding),
                                                filter_init=Normal(0.01),
                                           name=name
            ))
        elif type == 'CBN':
            add_layer(net, ConvBatchNormalization(shape=[act_channels()], name=name))
        elif type == 'REC':
            add_layer(net, ApplyActivation(rectify, name=name))
        elif type == 'MON':
            add_layer(net, ActivationMonitoring(name=name))
        elif type == 'MAXP':
            add_layer(net, MaxPoolingCudnn(act_h(), act_w(),
                                                pooling_size=(layer_cfg.pooling_size),
                                                pooling_stride=(layer_cfg.pooling_stride),
                                                padding=(layer_cfg.padding),
                                                name=name))
            # add_layer(net, MaxPooling(act_h(), act_w(), pooling_size=layer_cfg.pooling_size,
            # pooling_stride=layer_cfg.pooling_stride, name=name))
        else:
            print 'RRRRRRRRRRRRRRRr'
            raise RuntimeError()

    return net

def create_conv_column(cfg, prefix=''):
    net = Sequential(name=prefix + 'sequential')
    counter = {'CONV': 0, 'CBN': 0, 'REC': 0, 'MON': 0, 'MAXP': 0}
    print 'len', len(cfg)
    for layer_cfg in cfg:
        type = layer_cfg.type
        print 'create', type

        counter[type] += 1
        name = prefix + type + str(counter[type])

        if type == 'CONV':
            add_layer(net, CudaConv2dCudnn(act_h(), act_w(),
                                                workmem='small',
                                                n_input_channels=act_channels(),
                                                n_filters=layer_cfg.n_filters,
                                                kernel_size=(layer_cfg.kernel_size),
                                                kernel_stride=(layer_cfg.kernel_stride),
                                                padding=(layer_cfg.padding),
                                                filter_init=Uniform(
                                                    range=w(layer_cfg.kernel_size ** 2 * act_channels())),
                                                filter_bias_init=Uniform(
                                                    range=w(layer_cfg.kernel_size ** 2 * act_channels()),
                                                ),
                                           name=name
            ))
        elif type == 'CBN':
            add_layer(net, ConvBatchNormalization(shape=[act_channels()], name=name))
        elif type == 'REC':
            add_layer(net, ApplyActivation(rectify, name=name))
        elif type == 'MON':
            add_layer(net, ActivationMonitoring(name=name))
        elif type == 'MAXP':
            add_layer(net, MaxPoolingCudnn(act_h(), act_w(),
                                                pooling_size=(layer_cfg.pooling_size),
                                                pooling_stride=(layer_cfg.pooling_stride),
                                                padding=(layer_cfg.padding),
                                                name=name))
            # add_layer(net, MaxPooling(act_h(), act_w(), pooling_size=layer_cfg.pooling_size,
            # pooling_stride=layer_cfg.pooling_stride, name=name))
        else:
            print 'RRRRRRRRRRRRRRRr'
            raise RuntimeError()

    return net

import theano.tensor as T
def create_very_leaky(a):
    def very_leaky(x):
        return T.maximum(x, a * x)

    return very_leaky

def create_conv_colum(image_size, prefix, convs, untie_biases=False, activation_fun=relu):
        features = Sequential(name='features')
        input_size = image_size
        print 'input_size__ {}'.format(input_size)
        i = 0

        channels = 3
        for n_filters, do_pool, ac_mon in convs:

            conv_layer = CudaConv2dCudnn(
                input_height=input_size[0],
                input_width=input_size[1],
                n_input_channels=channels,
                n_filters=n_filters,
                kernel_size=(3, 3),
                kernel_stride=(1, 1),
                padding=(1, 1),
                filter_init=Normal(0.01),
                name=prefix + 'conv{}'.format(i),
                untie_biases=untie_biases
            )

            features.add(conv_layer)
            input_size = (conv_layer.get_output_height(), conv_layer.get_output_width())
            features.add(ConvBatchNormalization(n_filters, name=prefix + 'conv_batch_norm{}'.format(i)))
            features.add(ApplyActivation(activation_fun))

            if do_pool:
                pool_layer = MaxPoolingCudnn(
                    input_height=input_size[0],
                    input_width=input_size[1],
                    pooling_size=(3, 3),
                    pooling_stride=(2, 2),
                    padding=(1, 1),
                    name=prefix + 'pool{}'.format(i)
                )
                features.add(pool_layer)
                input_size = (pool_layer.get_output_height(), pool_layer.get_output_width())
            if ac_mon:
                features.add(ActivationMonitoring(name=prefix + 'acmon{}'.format(i)))
            i += 1
            print 'input_size {}'.format(input_size)
            channels = n_filters
        return features, input_size




