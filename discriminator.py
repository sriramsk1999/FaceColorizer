import tensorflow as tf
import ops


class Discriminator(object):
    def __init__(self, name, is_train, norm='instance', activation='leaky'):
        print('Init Discriminator '+name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._reuse = False

    def __call__(self, input):
        with tf.compat.v1.variable_scope(self.name, reuse=self._reuse):
            D = ops.conv_block(input, 64, 'C64', 4, 2, self._is_train,
                               self._reuse, norm=None, activation=self._activation)
            D = ops.conv_block(D, 128, 'C128', 4, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            D = ops.conv_block(D, 256, 'C256', 4, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            D = ops.conv_block(D, 512, 'C512', 4, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            D = ops.conv_block(D, 1, 'C1', 4, 1, self._is_train,
                               self._reuse, norm=None, activation=None, bias=True)
            D = tf.reduce_mean(D, axis=[1,2,3])

            self._reuse = True
            self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return D
