class namescope(object):
    namescopes = []

    def __init__(self,
                 name=None,
                 abs_name=None,
                 initializer=None,
                 regularizer=None,
                 reuse=None,
                 dtype=None):
        self.name = name
        prefix = "%s/%s" % ('/'.join(_.name
                                     for _ in namescope.namescopes), name)
        self.prefix = prefix if not abs_name else abs_name
        self.initializer = initializer
        self.regularizer = regularizer
        self.reuse = reuse
        self.dtype = dtype

    def namescope(self):
        return self

    def gen_name(self, name):
        return '%s/%s' % (self.prefix, name)

    def __enter__(self):
        namescope.namescopes.append(self)

    def __exit__(self, type, value, traceback):
        namescope.names.pop()

    def __str__(self):
        return self.prefix

    def __repr__(self):
        return "<namescope %s>" % self.__str__()


class Block(object):
    def __init__(self):
        # NOTE if block has its own scope, namescope is not needed to inherit
        # parent's namescope
        # or namescope should be a static member of Block
        self.namescope = namescope()


g_block = Block()


class Variable(object):
    counter = 0

    def __init__(self,
                 name=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 reuse=None,
                 block=None):
        if not name:
            name = "var-%d" % Variable.counter
            Variable.counter += 1
        if not block:
            block = g_block
        # NOTE need to check unique
        self.block = block
        self.name = block.namescope.gen_name(name)
        # set initializer
        if not initializer and block.namescope.initializer:
            initializer = block.namescope.initializer
        self.initializer = initializer
        # set regularizer
        if not regularizer and block.namescope.regularizer:
            regularizer = block.namescope.regularizer
        self.regularizer = regularizer
        # set reuse
        if reuse is None and block.namescope.reuse is not None:
            reuse = block.namescope.reuse
        self.reuse = reuse
        # set trainable


class Operator(object):
    counter = 0

    def __init__(self, type):
        '''
        each op has a unique name, two op with the same prefix will be put
        in the same group for visulazation.
        '''
        self.name = Block.namescope.gen_name('op-%d' % counter)
        Operator.counter += 1


def fc(inputs,
       num_outputs,
       activation,
       reuse=false,
       trainable=True,
       weights_initializer,
       biases_initializer,
       namescope=None):
    '''
    Args:

      inputs: A tensor of at least rank 2.
      num_outputs: int, the number of output units in the layer.
      activation: activation function, default is sigmoid.
      weights_initializer: an initializer for the weights.
      biases_initializer: an initializer for the biases, if None skip biases.
      reuse: Whether or not the layer and its parameters should be reused. To be
        able to reuse the layer, namescope should be given.
      trainable: if True, the parameters in this layer will be updated during training.

    Returns:
       the variable representing the result of the series of operations.
    '''
    W_shape = [x.shape[1], size]
    b_shape = [x.shape[1], 1]
    W_initializer = weights_initializer if weights_initializer else pd.gaussian_random_initializer(
    )
    b_initializer = biases_initializer if weights_initializer else pd.zeros_initializer(
    )

    if not reuse:
        # make unique names for both W and b
        W = pd.Variable(shape=W_shape, initializer=W_initializer)
        b = pd.Variable(shape=b_shape, initializer=b_initializer)

    else:
        assert namescope, "namescope should be provided to help reuse the parameters"
        with namescope:
            W = pd.Variable(
                name="W", shape=W_shape, initializer=W_initializer, reuse=True)
            b = pd.Variable(
                name="b", shape=b_shape, initializer=b_initializer, reuse=True)
    return pd.add_two(pd.mat_mul(W, x), b)


if __name__ == '__main__':
    import paddle as pd

    with pd.namescope(initializer=pd.gaussian_random_initializer()):
        a1 = pd.Variable(shape=[20, 20])
        a2 = pd.Variable(shape=[20, 20])
        a3 = pd.Variable(shape=[20, 20])
        b = pd.Variable(shape=[20, 20])

    def two_level_fc(x, prefix=None):
        fc_out1 = fc(x,
                     namescope=pd.namescope(prefix + '-level0')
                     if prefix else None)
        fc_out2 = fc(fc_out1,
                     namescope=pd.namescope(prefix + '-level1')
                     if prefix else None)
        return fc_out2

    # the first 3 fcs share parameter variables with namescope "fc0"
    # while the last fc use its own parameters.
    fc_out1 = two_level_fc(a1, "fc0")
    fc_out2 = two_level_fc(a2, "fc0")
    fc_out3 = two_level_fc(a3, "fc0")
    fc_out4 = two_level_fc(a4)
