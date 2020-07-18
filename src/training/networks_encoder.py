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

# apply a individual dense layer to each channel of the input (along axis=1)
# input: w of shape [batch, num_channels, input_filters]
# output of shape [batch, num_channels, output_filters]
def channel_independent_dense(x, output_filters, num_channels):
    batch_size =  tf.shape(x)[0];
    
    output_list = []
    for i in range(num_channels):
        x_i = x[:,i, :]
        x_i = dense(x_i, fmaps=output_filters, gain=1, use_wscale=False)
        x_i = tf.reshape(x_i, [batch_size, 1, output_filters])
        output_list.append(x_i)
    return tf.concat(output_list, axis=1)


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


def Encoder(embedded_w, input_keypoints, size=128, filter=64, filter_max=512, num_layers=12, phase=True, **kwargs):
    #print('using bn encoder phase: ', phase)
    s0 = 4
    num_blocks = int(np.log2(size / s0))

    # define input shapes for the network
    # todo: aktuell am imput img nix verändert!!!
    embedded_w.set_shape([None, num_layers, 512])
    
    batch_size =  tf.shape(embedded_w)[0];
    
    input_keypoints.set_shape([None, 69])

    #input_concatenated = tf.concat((input_img, input_landmarks), axis=1) # [0: batch, 1: channels, 2,3: hw]

    with tf.variable_scope('encoder'):
        with tf.variable_scope('landmark_encoder_fc'):
            lm_context= dense(input_keypoints, 160, gain=1, use_wscale=False)
            lm_context = tf.reshape(lm_context, [batch_size, 1, 160])
            lm_context = tf.tile(lm_context, [1,12,1])
        
        with tf.variable_scope('latent_code_encoder'):
            w_context = channel_independent_dense(embedded_w, 32, num_layers)
            w_context = leaky_relu(bn(w_context, phase=phase, name='bn_latent_code_encoder'))
            
        concatenated_context = tf.concat((w_context, lm_context), axis=2)
        
        with tf.variable_scope('decoder'):
            latent_modifier = channel_independent_dense(concatenated_context, 512, num_layers)
            latent_modifier = bn(latent_modifier, phase=phase, name='bn_decoder')
        
        latent_w = tf.math.add(embedded_w, latent_modifier)
        
        return latent_w