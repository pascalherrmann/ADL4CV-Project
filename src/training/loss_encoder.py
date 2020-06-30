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
def E_loss(E, G, D, perceptual_model, real_portraits, shuffled_portraits, real_landmarks, shuffled_landmarks, training_mode, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256):

    '''
    # Only needed for Alternating Training
    with tf.device("/cpu:0"):
        appearance_flag = tf.math.equal(training_mode, "appearance")

    reals = tf.cond(appearance_flag, lambda: real_portraits, lambda: shuffled_portraits)
    '''


    '''
    reals = tf.case(
                [(tf.math.equal(training_mode, "appearance"), lambda: real_portraits), 
                 (tf.math.equal(training_mode, "pose"), lambda: shuffled_portraits)],
                  default = lambda: real_portraits*0, exclusive=True)
    '''



    '''
    # CYCLE CONSISTENCY
    * 1: Feed Original Portrait + SHUFFLED Landmark into Encoder -> Get W for "Manipulated Image"
    * 2: Feed Manipulated Image + SHUFFLED Landmark into Cond. Discriminator -> Fake scores manpulated
    * 3: Then: Feed Manipulated Image + Original Landmark into Encoder -> Get W for "Reconstructed Image"
    * 4: Feed "Reconstructed Image" + ORIGINAL Landmark into Cond. Discriminator -> reconstructed image
    * 5: + Take Reconsturction & Perceptual Loss between Reconstructed & Original Image


    Davor haben wir immer shuffled portraits übergeben. Jetzt brauchen wir aber: Shuffled Landmarks....
    '''
    reals = real_portraits
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]


    # direct reconstruction
    w_direct_rec = E.get_output_for(real_portraits, real_landmarks, phase=True)
    w_direct_rec_tensor = tf.reshape(w_direct_rec, [reals.shape[0], num_layers, latent_dim])
    img_direct_rec = G.components.synthesis.get_output_for(w_direct_rec_tensor, randomize_noise=False)
    direct_fake_scores_out = fp32(D.get_output_for(img_direct_rec, real_landmarks, None))


    # 1
    w_manipulated = E.get_output_for(real_portraits, shuffled_landmarks, phase=True)
    w_manipulated_tensor = tf.reshape(w_manipulated, [reals.shape[0], num_layers, latent_dim])
    img_manipulated = G.components.synthesis.get_output_for(w_manipulated_tensor, randomize_noise=False)

    # 2
    manipulated_fake_scores_out = fp32(D.get_output_for(img_manipulated, shuffled_landmarks, None))

    # 3
    w_reconstructed = E.get_output_for(img_manipulated, real_landmarks, phase=True)
    w_reconstructed_tensor = tf.reshape(w_reconstructed, [reals.shape[0], num_layers, latent_dim])
    img_reconstructed = G.components.synthesis.get_output_for(w_reconstructed_tensor, randomize_noise=False)

    # 4
    reconstructed_fake_scores_out = fp32(D.get_output_for(img_reconstructed, real_landmarks, None))

    # 5
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

        #
        # Direct Rec Loss
        #
        vgg16_input_fake_direct = tf.transpose(img_direct_rec, perm=[0, 2, 3, 1])
        vgg16_input_fake_direct = tf.image.resize_images(vgg16_input_fake_direct, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_fake_direct = ((vgg16_input_fake_direct + 1) / 2) * 255
        vgg16_feature_fake_direct = perceptual_model(vgg16_input_fake_direct)
        recon_loss_feats_direct = feature_scale * tf.reduce_mean(tf.square(vgg16_feature_real - vgg16_feature_fake_direct))

        # recon
        recon_loss_pixel_direct = tf.reduce_mean(tf.square(img_direct_rec - real_portraits))
        recon_loss_feats_direct = autosummary('Loss/scores/loss_feats_direct', recon_loss_feats_direct)
        recon_loss_pixel_direct = autosummary('Loss/scores/loss_pixel_direct', recon_loss_pixel_direct)
        recon_loss_direct = recon_loss_feats_direct + recon_loss_pixel_direct
        recon_loss_direct = autosummary('Loss/scores/recon_loss', recon_loss_direct)

    with tf.variable_scope('adv_loss'):
        D_scale = autosummary('Loss/scores/d_scale', D_scale)
        adv_loss_manipulated = D_scale * tf.reduce_mean(tf.nn.softplus(-manipulated_fake_scores_out))
        adv_loss_manipulated = autosummary('Loss/scores/adv_loss_manipulated', adv_loss_manipulated)
        
        adv_loss_reconstructed = D_scale * tf.reduce_mean(tf.nn.softplus(-reconstructed_fake_scores_out))
        adv_loss_reconstructed = autosummary('Loss/scores/adv_loss_reconstructed', adv_loss_reconstructed)

        adv_loss_direct = D_scale * tf.reduce_mean(tf.nn.softplus(-direct_fake_scores_out))
        adv_loss_direct = autosummary('Loss/scores/adv_loss_direct', adv_loss_direct)

    '''
    loss = tf.cond(appearance_flag, lambda: adv_loss + recon_loss, lambda: adv_loss)
    '''

    loss = (adv_loss_manipulated + adv_loss_reconstructed + adv_loss_direct) + recon_loss + 0.5 * recon_loss_direct

    '''
    loss = tf.case(
                [(tf.math.equal(training_mode, "appearance"), lambda: adv_loss), 
                 (tf.math.equal(training_mode, "pose"), lambda: adv_loss + recon_loss)],
                  default = lambda: adv_loss*0, exclusive=True)
    '''

    return loss, (recon_loss+recon_loss_direct), adv_loss_manipulated

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_logistic_simplegp(E, G, D, real_portraits, shuffled_portraits, real_landmarks, training_mode, r1_gamma=10.0):

    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]

    with tf.device("/cpu:0"):
            appearance_flag = tf.math.equal(training_mode, "appearance")

    portraits = tf.cond(appearance_flag, lambda: real_portraits, lambda: shuffled_portraits)

    '''
    portraits = tf.case(
                [(tf.math.equal(training_mode, "appearance"), lambda: real_portraits), 
                 (tf.math.equal(training_mode, "pose"), lambda: shuffled_portraits)],
                  default = lambda: real_portraits*0, exclusive=True)
    '''

    latent_w = E.get_output_for(portraits, real_landmarks, phase=True)
    latent_wp = tf.reshape(latent_w, [portraits.shape[0], num_layers, latent_dim]) # make synthetic from shuffled ones!
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
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
