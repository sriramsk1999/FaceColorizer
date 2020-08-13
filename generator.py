import tensorflow as tf
import ops


class Generator(object):
    def __init__(self, name, is_train, norm='instance', activation='relu',
                 image_size=256):
        print('Init Generator '+name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._num_res_block = 9
        self._reuse = False

    def __call__(self, input):
        with tf.compat.v1.variable_scope(self.name, reuse=self._reuse):
            G = ops.conv_block(input, 32, 'c7s1-32', 7, 1, self._is_train,
                               self._reuse, self._norm, self._activation, pad='REFLECT')
            G = ops.conv_block(G, 64, 'd64', 3, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            G = ops.conv_block(G, 128, 'd128', 3, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            for i in range(self._num_res_block):
                G = ops.residual(G, 128, 'R128_{}'.format(i), self._is_train,
                                 self._reuse, self._norm)
            G = ops.deconv_block(G, 64, 'u64', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation)
            G = ops.deconv_block(G, 32, 'u32', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation)
            G = ops.conv_block(G, 3, 'c7s1-3', 7, 1, self._is_train,
                               self._reuse, norm=None, activation='tanh', pad='REFLECT')

            self._reuse = True
            self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return G
