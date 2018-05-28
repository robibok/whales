import copy
from TheanoLib.init import Normal
from TheanoLib.modules import Sequential, Flatten, Dropout, Dense, identity, Softmax, FanOut, Parallel, Subtensor, \
    SimpleApply, softmax
from architecture import create_conv_colum
import theano.tensor as T

def create(image_size=(448, 448), n_outs=[447], dropout=False,
        fc_l2_reg=None, conv_l2_reg=None, **kwargs):
    print '... building the model'
    print 'image_size', image_size, kwargs
    classifier = Sequential(name='classifier')
    net = Sequential(name='sequential')

    convs = [(32, 1, 0),  (64, 1, 0), (64, 0, 0), (128, 0, 0), (128, 1, 0), (256, 0, 0), (256, 1, 0),
             (256, 0, 0), (256, 1, 0), (256, 0, 0), (256, 1, 0)]

    features, size1 = create_conv_colum(image_size, 'MAIN.', convs)

    net.add(features)
    classifier.add(Flatten())

    if dropout:
        classifier.add(Dropout(p_of_zero=dropout))

    def f(input):
        outs = []
        s = 0
        for n_out in n_outs:
            outs.append(softmax(input[:, s: s + n_out]))
            s += n_out

        return T.concatenate(outs, axis=1)

    classifier.add(Dense(
        n_input=convs[-1][0] * size1[0] * size1[1],
        n_output=sum(n_outs),
        nonlinearity=identity,
        W_init=Normal(0.001),
        name='dense'
    ))


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