import tensorflow as tf


def _norm(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.compat.v1.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            mean, sigma = tf.nn.moments(input, [1, 2], keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + eps)
            out = normalized
    elif norm == 'batch':
        with tf.compat.v1.variable_scope('batch_norm', reuse=reuse):
            out = tf.contrib.layers.batch_norm(input,
                                               decay=0.99, center=True,
                                               scale=True, is_training=is_train,
                                               updates_collections=None)
    else:
        out = input

    return out

def _activation(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.2)(input)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    else:
        return input

def conv2d(input, num_filters, filter_size, stride, reuse=False,
           pad='SAME', dtype=tf.float32, bias=False):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]

    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if pad == 'REFLECT':
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0,0],[p,p],[p,p],[0,0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)

    if bias:
        b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
        conv = conv + b
    return conv

def conv2d_transpose(input, num_filters, filter_size, stride, reuse,
                     pad='SAME', dtype=tf.float32):
    assert pad == 'SAME'
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, c]
    output_shape = [n, h * stride, w * stride, num_filters]

    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)
    return deconv

def conv_block(input, num_filters, name, k_size, stride, is_train, reuse, norm,
          activation, pad='SAME', bias=False):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        out = conv2d(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out

def residual(input, num_filters, name, is_train, reuse, norm, pad='REFLECT'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        with tf.compat.v1.variable_scope('res1', reuse=reuse):
            out = conv2d(input, num_filters, 3, 1, reuse, pad)
            out = _norm(out, is_train, reuse, norm)
            out = tf.nn.relu(out)

        with tf.compat.v1.variable_scope('res2', reuse=reuse):
            out = conv2d(out, num_filters, 3, 1, reuse, pad)
            out = _norm(out, is_train, reuse, norm)

        return tf.nn.relu(input + out)

def deconv_block(input, num_filters, name, k_size, stride, is_train, reuse, norm, activation):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        out = conv2d_transpose(input, num_filters, k_size, stride, reuse)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out
