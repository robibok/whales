import copy
from TheanoLib.init import Normal
from TheanoLib.modules import Sequential, Flatten, Dropout, Dense, identity, Softmax, FanOut, Parallel, Subtensor, \
    SimpleApply, softmax, CudaConv2dCudnn, ApplyActivation, ConvBatchNormalization, MaxPoolingCudnn, rectify, Reshape, \
    softmax3
from whales.architecture import create_conv_colum
import theano.tensor as T

def create(image_size=(448, 448), n_outs=None, dropout=False,
                 fc_l2_reg=None, conv_l2_reg=None, **kwargs):
    n_first = 16
    n_fc = 64
    print '... building the model'
    features = Sequential(name='features')
    classifier = Sequential(name='classifier')
    net = Sequential(name='sequential')

    input_size = image_size
    print 'input_size {}'.format(input_size)
    i = 0
    while input_size[0] >= 12:
        if i == 0:
            channels = 3
            n_filters = n_first
        elif i == 1:
            channels = n_first
            n_filters = n_fc
        else:
            channels = n_fc
            n_filters = n_fc

        conv_layer = CudaConv2dCudnn(
            input_height=input_size[0],
            input_width=input_size[1],
            n_input_channels=channels,
            n_filters=n_filters,
            kernel_size=(3, 3),
            kernel_stride=(1, 1),
            padding=(1, 1),
            filter_init=Normal(0.01),
            name='conv{}'.format(i)
        )

        features.add(conv_layer)
        input_size = (conv_layer.get_output_height(), conv_layer.get_output_width())
        features.add(ConvBatchNormalization(n_filters, name='conv_batch_norm{}'.format(i)))
        features.add(ApplyActivation(rectify))

        pool_layer = MaxPoolingCudnn(
            input_height=input_size[0],
            input_width=input_size[1],
            pooling_size=(3, 3),
            pooling_stride=(2, 2),
            padding=(1, 1),
            name='pool{}'.format(i)
        )
        features.add(pool_layer)
        input_size = (pool_layer.get_output_height(), pool_layer.get_output_width())
        i += 1
        print 'input_size {}'.format(input_size)

    net.add(features)
    classifier.add(Flatten())
    # classifier.add(Dense(
    #     n_input=n_fc * input_size[0] * input_size[1],
    #     n_output=n_fc,
    #     nonlinearity=rectify,
    #     W_init=Normal(0.001),
    #     name='dense1'
    # ))
    classifier.add(Dense(
        n_input=n_fc * input_size[0] * input_size[1],
        n_output=sum(n_outs),
        nonlinearity=identity,
        W_init=Normal(0.001),
        name='dense2'
    ))

    def f(input):
        outs = []
        s = 0
        for n_out in n_outs:
            outs.append(softmax(input[:, s: s + n_out]))
            s += n_out

        return T.concatenate(outs, axis=1)

    classifier.add(SimpleApply(f))
    net.add(classifier)


    ##########
    arch = copy.deepcopy(net)

    print 'Calling allocate_params()'
    net.allocate_params()

    print 'Calling initialize_params()'
    net.initialize_params()


    reg_params = (zip(classifier.get_reg_params(), len(classifier.get_reg_params()) * [fc_l2_reg]) +
                  zip(features.get_reg_params(), len(features.get_reg_params()) * [conv_l2_reg]))

    return arch, net, reg_params
