import tensorflow as tf
import numpy as np


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02)) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))


def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


# convert w to l:
# input: w of shape [batch, 12, 1, 512]
# weight-matrix of shape [12, 512, 32]
# output of shape [batch, 12, 1, 32]
def linear_independent_w_to_l(x):
    w = get_weight([12, 512, 32], gain=1, use_wscale=False)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

# p: shape [batch, 160]
# l: shape [batch, 12, 1, 32]
def linear_independent_p_l_to_diff(x):
    w = get_weight([12, 192, 512], gain=1, use_wscale=False)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')


def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.constant_initializer(0.0))
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])


def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)



def bn(x, phase, name='batch_norm'):
    with tf.variable_scope(name):
        x = tf.contrib.layers.batch_norm(x,
                                         decay=0.9,
                                         center=True,
                                         scale=False,
                                         epsilon=1e-5,
                                         updates_collections=None,
                                         is_training=phase,
                                         data_format='NCHW')
        return x


def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]), 1, np.int32(s[3]), 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]) * factor, np.int32(s[3]) * factor])
        return x


def _shortcut(x, fout, learned_shortcut):
    if learned_shortcut:
        x_s = conv2d(x, fmaps=fout, kernel=1, use_wscale=False)
    else:
        x_s = x
    return x_s


def residual_block_bn(inputs, fin, fout, phase, scope): #resnet v1
    hidden = min(fin, fout)
    learned_ = (fin != fout)
    with tf.variable_scope(scope):
        with tf.variable_scope('shortcut'):
            x_shortcut = _shortcut(inputs, fout, learned_shortcut=learned_)
            if learned_:
                x_shortcut = leaky_relu(bn(x_shortcut, phase=phase, name='shortcut_bn'))
        with tf.variable_scope('conv1'):
            net = conv2d(inputs, fmaps=hidden, kernel=3, use_wscale=False)
            net = leaky_relu(bn(net, phase=phase, name='bn_1'))
        with tf.variable_scope('conv2'):
            net = conv2d(net, fmaps=fout, kernel=3, use_wscale=False)
            net = leaky_relu(bn(net, phase=phase, name='bn_2'))
        net = net + x_shortcut

    return net


def Encoder(input_lm, size=128, filter=64, filter_max=512, num_layers=12, phase=True, **kwargs):
    #print('using bn encoder phase: ', phase)
    s0 = 4
    num_blocks = int(np.log2(size / s0)) + 2

    # define input shapes for the network
    # todo: aktuell am imput img nix verändert!!!
    input_lm.set_shape([None, 3, size, size])

    input_concatenated = input_lm

    with tf.variable_scope('encoder'):
        with tf.variable_scope('input_image_stage'):
            net = conv2d(input_concatenated, fmaps=filter, kernel=3, use_wscale=False)
            net = leaky_relu(bn(net, phase=phase, name='bn_input_stage'))

        for i in range(num_blocks):
            name_scope = 'encoder_res_block_%d' % (i)
            nf1 = min(filter * 2 ** i, filter_max)
            nf2 = min(filter * 2 ** (i + 1), filter_max)
            net = downscale2d(net, factor=2)
            net = residual_block_bn(net, fin=nf1, fout=nf2, phase=phase, scope=name_scope)

        with tf.variable_scope('encoder_fc'):
            latent_w = dense(net, fmaps=160, num_layers, gain=1, use_wscale=False)
            latent_w = bn(latent_w, phase=phase, name='fc_1')

        return latent_w # landmark embedding p of shape 180


def Stylerig_Encoder(input_w, size=128, filter=64, filter_max=512, num_layers=12, phase=True, **kwargs):
    input_w.set_shape([None, 12, 512])
    input_w = tf.reshape(input_w, [input_w.shape[0], 12, 1, 512])

    with tf.variable_scope('stylerig_encoder'):
        with tf.variable_scope('stylerig_encoder_fc'):
            latent_w = linear_independent_w_to_l(input_w):
        return latent_w


def Stylerig_Decoder(input_l, input_p):

    input_l.set_shape([None, 12, 1, 32])
    input_p.set_shape([None, 160])
    input_p = tf.reshape(input_p, [input_p.shape[0], 1, 1, 160])
    input_p = tf.tile(input_p, [1,12,1,1])
    input = tf.concat([input_l, input_p], axis=3)

    with tf.variable_scope('stylerig_decoder'):
        with tf.variable_scope('stylerig_decoder_fc'):
            # concatenate l and w
            w_diff = get_matrix(input_p, 512, 12, gain=np.sqrt(2), use_wscale=False):
        return w_diff
