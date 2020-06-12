"""Loss functions for training encoder."""

import numpy as np
import tensorflow as tf
from dnnlib.tflib.autosummary import autosummary


#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


#----------------------------------------------------------------------------
# Encoder loss function: takes img -> loss: recon (px + ft) + adv (discriminator for fakes)
def E_loss(E, G, D, perceptual_model, reals, real_landmarks, feature_scale=0.00005, D_scale=1.0, perceptual_img_size=256):

    # get dimensions of latent space w (512*12)
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]

    # feed images (for now: only landmarks) into ENCODER! (stays for now the same, just with landmarks) -> get latent code w
    latent_w = E.get_output_for(real_landmarks, phase=True)
    
    #for the appearance, random sample in z-space and map to w
    latent_w_appearance = G.components.mapping.get_output_for(np.random.randn(512))
    
    #combine landmark and appearance latent_codes
    latent_w += latent_w_appearance

    # reshape fully connected latent_w to [batch, num_layers]
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])

    # put w's in generator to get synthetic reconstructions
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)

    # NEW: feed original landmarks + fake image in discriminator!
    fake_scores_out = fp32(D.get_output_for(fake_X, real_landmarks, None))

    with tf.variable_scope('adv_loss'):
        D_scale = autosummary('Loss/scores/d_scale', D_scale)
        adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_scores_out))# * D_scale
        adv_loss = autosummary('Loss/scores/adv_loss', adv_loss)

    loss = adv_loss

    return loss, 0, adv_loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_logistic_simplegp(E, G, D, reals, real_landmarks, r1_gamma=10.0):

    # get shape of w
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]

    # feed real images in encoder to get w
    latent_w = E.get_output_for(real_landmarks, phase=True)
    
    #for the appearance, random sample in z-space and map to w
    latent_w_appearance = G.components.mapping.get_output_for(np.random.randn(512))
    
    #combine landmark and appearance latent_codes
    latent_w += latent_w_appearance

    # reshape w
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])

    # feed w in generator to "reconstruct" image (i.e., get fake images)
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)

    # get scores for fake and real images
    # NEW: also feed in landmarks
    real_scores_out = fp32(D.get_output_for(reals, real_landmarks, None))
    fake_scores_out = fp32(D.get_output_for(fake_X, real_landmarks, None))

    #real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    #fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss_fake = tf.reduce_mean(tf.nn.softplus(fake_scores_out))
    loss_real = tf.reduce_mean(tf.nn.softplus(-real_scores_out))
    
    loss_real = autosummary('Loss/scores/real', loss_real)
    loss_fake = autosummary('Loss/scores/fake', loss_fake)

    with tf.name_scope('R1Penalty'):
        real_grads = fp32(tf.gradients(real_scores_out, [reals])[0])
        r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
        r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss_gp = r1_penalty * (r1_gamma * 0.5)
    loss = loss_fake + loss_real + loss_gp
    return loss, loss_fake, loss_real, loss_gp
