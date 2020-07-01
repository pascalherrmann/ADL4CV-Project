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
# Encoder loss function .
def E_loss(E, G, D, E_lm, E_rig, Dec_rig, perceptual_model, real_portraits, shuffled_portraits, real_landmarks, shuffled_landmarks, training_mode, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256):

    '''
    Rignet - First Try:

    1. Take source image and embedd it into W using pre-trained vanilla-encoder. => w [12*512]
    2. Rignet-Encoder: Take w and embedd into l => [12*32]
    3. Take target landmarks and embedd into p => p [1x160]
    4. Rigner-Decoder: take p, l -> gt diff [12*512]
    5. add to w.

    then:
    6. feed in generator

    We need:
    * Encoder Pretrained
    * Encoder_LM for Landmarks, new!!!
    * Matrix w_to_l  12*512*32
    * Matrix pl_to_diff 12*(32+160)*512
    '''

    # 1
    w = E.get_output_for(real_portraits, phase=True)
    w = tf.reshape(w, [real_portraits.shape[0], 12, 512])

    # 2
    l = E_rig.get_output_for(w)

    # 3
    p = E_lm.get_output_for(shuffled_landmarks)

    # 4
    diff = Dec_rig.get_output_for(l, p)
    diff = tf.reshape(diff, [real_portraits.shape[0], 12, 512])


    # 5
    w_manipulated_tensor = w + diff

    # w should be of shape [batch, 12, 512]
    # diff should be of shape:


    #
    img_manipulated = G.components.synthesis.get_output_for(w_manipulated_tensor, randomize_noise=False)
    manipulated_fake_scores_out = fp32(D.get_output_for(img_manipulated, shuffled_landmarks, None))


    ##
    #
    # Then: Cycle consistency. Map everything back.
    #
    ##

    l_manipulated= E_rig.get_output_for(w_manipulated_tensor)
    p_original = E_lm.get_output_for(real_landmarks)
    diff_cycle =  Dec_rig.get_output_for(l_manipulated, p_original)
    diff_cycle = tf.reshape(diff_cycle, [real_portraits.shape[0], 12, 512])
    w_reconstructed = w_manipulated_tensor + diff_cycle
    w_reconstructed_tensor = tf.reshape(w_reconstructed, [real_portraits.shape[0], 12, 512])
    img_reconstructed = G.components.synthesis.get_output_for(w_reconstructed_tensor, randomize_noise=False)
    reconstructed_fake_scores_out = fp32(D.get_output_for(img_reconstructed, real_landmarks, None))

    with tf.variable_scope('recon_loss'):
        # feature
        vgg16_input_real = tf.transpose(real_portraits, perm=[0, 2, 3, 1])
        vgg16_input_real = tf.image.resize_images(vgg16_input_real, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_real = ((vgg16_input_real + 1) / 2) * 255

        vgg16_input_fake = tf.transpose(img_reconstructed, perm=[0, 2, 3, 1])
        vgg16_input_fake = tf.image.resize_images(vgg16_input_fake, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_fake = ((vgg16_input_fake + 1) / 2) * 255
        vgg16_feature_real = perceptual_model(vgg16_input_real)
        vgg16_feature_fake = perceptual_model(vgg16_input_fake)
        recon_loss_feats = feature_scale * tf.reduce_mean(tf.square(vgg16_feature_real - vgg16_feature_fake))

        # recon
        recon_loss_pixel = tf.reduce_mean(tf.square(img_reconstructed - real_portraits))
        recon_loss_feats = autosummary('Loss/scores/loss_feats', recon_loss_feats)
        recon_loss_pixel = autosummary('Loss/scores/loss_pixel', recon_loss_pixel)
        recon_loss = recon_loss_feats + recon_loss_pixel
        recon_loss = autosummary('Loss/scores/recon_loss', recon_loss)

    with tf.variable_scope('adv_loss'):
        D_scale = autosummary('Loss/scores/d_scale', D_scale)
        adv_loss_manipulated = D_scale * tf.reduce_mean(tf.nn.softplus(-manipulated_fake_scores_out))
        adv_loss_manipulated = autosummary('Loss/scores/adv_loss_manipulated', adv_loss_manipulated)
        
        adv_loss_reconstructed = D_scale * tf.reduce_mean(tf.nn.softplus(-reconstructed_fake_scores_out))
        adv_loss_reconstructed = autosummary('Loss/scores/adv_loss_reconstructed', adv_loss_reconstructed)
    '''
    loss = tf.cond(appearance_flag, lambda: adv_loss + recon_loss, lambda: adv_loss)
    '''

    loss = adv_loss_manipulated + adv_loss_reconstructed + 2 * recon_loss

    '''
    loss = tf.case(
                [(tf.math.equal(training_mode, "appearance"), lambda: adv_loss), 
                 (tf.math.equal(training_mode, "pose"), lambda: adv_loss + recon_loss)],
                  default = lambda: adv_loss*0, exclusive=True)
    '''

    return loss, recon_loss, adv_loss_manipulated

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_logistic_simplegp(E, G, D, E_lm, E_rig, Dec_rig, real_portraits, shuffled_portraits, real_landmarks, training_mode, r1_gamma=10.0):

    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]

    # generate fakes
    # 1
    w = E.get_output_for(shuffled_portraits, phase=True)
    w = tf.reshape(w, [real_portraits.shape[0], 12, 512])

    # 2
    l = E_rig.get_output_for(w)
    # 3
    p = E_lm.get_output_for(real_landmarks)
    # 4
    diff = Dec_rig.get_output_for(l, p)
    diff = tf.reshape(diff, [real_portraits.shape[0], 12, 512])
    # 5
    w_manipulated = w + diff
    #
    w_manipulated_tensor = tf.reshape(w_manipulated, [real_portraits.shape[0], num_layers, latent_dim])
    fake_X = G.components.synthesis.get_output_for(w_manipulated_tensor, randomize_noise=False)

    real_scores_out = fp32(D.get_output_for(real_portraits, real_landmarks, None)) # real portraits, real landmarks
    fake_scores_out = fp32(D.get_output_for(fake_X, real_landmarks, None)) # synthetic portaits, real landmarks

    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss_fake = tf.reduce_mean(tf.nn.softplus(fake_scores_out))
    loss_real = tf.reduce_mean(tf.nn.softplus(-real_scores_out))

    loss_fake = autosummary('Loss/scores/loss_fake', loss_fake)
    loss_real = autosummary('Loss/scores/loss_real', loss_real)

    with tf.name_scope('R1Penalty'):
        real_grads = fp32(tf.gradients(real_scores_out, [real_portraits])[0])
        r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
        r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss_gp = r1_penalty * (r1_gamma * 0.5)
    loss = loss_fake + loss_real + loss_gp
    return loss, loss_fake, loss_real, loss_gp
