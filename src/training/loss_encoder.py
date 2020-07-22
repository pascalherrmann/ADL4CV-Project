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

def feedthrough(input_value):
    return input_value

def appearance_training(E, G, D, Inv, perceptual_model, real_portraits, shuffled_portraits, real_landmarks, shuffled_landmarks, training_flag, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256):
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    embedded_w = Inv.get_output_for(real_portraits, phase=True)
    embedded_w_tensor = tf.reshape(embedded_w, [real_portraits.shape[0], num_layers, latent_dim])
    
    latent_w = E.get_output_for(embedded_w_tensor, real_landmarks, phase=True)
    latent_wp = tf.reshape(latent_w, [real_portraits.shape[0], num_layers, latent_dim])
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
    fake_scores_out = fp32(D.get_output_for(fake_X, real_landmarks, None))
    
    with tf.variable_scope('recon_loss_appearance'):
        recon_loss = tf.reduce_mean(tf.square(latent_wp - embedded_w_tensor))
        recon_loss = autosummary('Loss/scores/recon_loss_appearance', recon_loss)

    with tf.variable_scope('adv_loss_appearance'):
        adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_scores_out))
        adv_loss = autosummary('Loss/scores/adv_loss_appearance', adv_loss)

    loss = adv_loss * D_scale  + recon_loss

    return loss, recon_loss, adv_loss

def pose_training(E, G, D, Inv, perceptual_model, real_portraits, shuffled_portraits, real_landmarks, shuffled_landmarks, training_flag, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256):
    '''
    # CYCLE CONSISTENCY
    * 1: Feed Original Portrait + SHUFFLED Landmark into Encoder -> Get W for "Manipulated Image"
    * 2: Feed Manipulated Image + SHUFFLED Landmark into Cond. Discriminator -> Fake scores manpulated
    * 3: Then: Feed Manipulated Image + Original Landmark into Encoder -> Get W for "Reconstructed Image"
    * 4: Feed "Reconstructed Image" + ORIGINAL Landmark into Cond. Discriminator -> reconstructed image
    * 5: + Take Reconsturction & Perceptual Loss between Reconstructed & Original Image
    '''
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    embedded_w = Inv.get_output_for(real_portraits, phase=True)
    embedded_w_tensor = tf.reshape(embedded_w, [real_portraits.shape[0], num_layers, latent_dim])
    # 1
    w_manipulated = E.get_output_for(embedded_w_tensor, shuffled_landmarks, phase=True)
    w_manipulated_tensor = tf.reshape(w_manipulated, [real_portraits.shape[0], num_layers, latent_dim])
    img_manipulated = G.components.synthesis.get_output_for(w_manipulated_tensor, randomize_noise=False)

    # 2
    manipulated_fake_scores_out = fp32(D.get_output_for(img_manipulated, shuffled_landmarks, None))

    # 3
    w_reconstructed = E.get_output_for(w_manipulated_tensor, real_landmarks, phase=True)
    w_reconstructed_tensor = tf.reshape(w_reconstructed, [real_portraits.shape[0], num_layers, latent_dim])
    img_reconstructed = G.components.synthesis.get_output_for(w_reconstructed_tensor, randomize_noise=False)

    # 4
    reconstructed_fake_scores_out = fp32(D.get_output_for(img_reconstructed, real_landmarks, None))

    # 5
    with tf.variable_scope('recon_loss_pose'):
        recon_loss = tf.reduce_mean(tf.square(w_reconstructed_tensor - embedded_w_tensor))
        recon_loss = autosummary('Loss/scores/recon_loss_pose', recon_loss)

    with tf.variable_scope('adv_loss_pose'):
        adv_loss_manipulated = tf.reduce_mean(tf.nn.softplus(-manipulated_fake_scores_out))
        adv_loss_manipulated = autosummary('Loss/scores/adv_loss_pose_manipulated', adv_loss_manipulated)
        
        adv_loss_reconstructed = tf.reduce_mean(tf.nn.softplus(-reconstructed_fake_scores_out))
        adv_loss_reconstructed = autosummary('Loss/scores/adv_loss_pose_reconstructed', adv_loss_reconstructed)


    loss = D_scale * adv_loss_manipulated + D_scale * adv_loss_reconstructed + recon_loss

    return loss, recon_loss, (adv_loss_manipulated + adv_loss_reconstructed)


#----------------------------------------------------------------------------
# Encoder loss function .
def E_loss(E, G, D, Inv, perceptual_model, real_portraits, shuffled_portraits, real_landmarks, shuffled_landmarks, training_flag, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256):

    with tf.device("/cpu:0"):
        appearance_flag = tf.math.equal(training_flag, "appearance")

    loss, recon_loss, adv_loss = tf.cond(appearance_flag, lambda: appearance_training(E, G, D, Inv, perceptual_model, real_portraits, shuffled_portraits, real_landmarks, shuffled_landmarks, training_flag, feature_scale, D_scale, perceptual_img_size), lambda: pose_training(E, G, D, Inv, perceptual_model, real_portraits, shuffled_portraits, real_landmarks, shuffled_landmarks, training_flag, feature_scale, D_scale, perceptual_img_size))

    return loss, recon_loss, adv_loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_logistic_simplegp(E, G, D, Inv, real_portraits, shuffled_portraits, real_landmarks, training_flag, r1_gamma=10.0):

    with tf.device("/cpu:0"):
        appearance_flag = tf.math.equal(training_flag, "appearance")
    
    portraits = tf.cond(appearance_flag, lambda: feedthrough(real_portraits), lambda: feedthrough(shuffled_portraits))
        
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    embedded_w = Inv.get_output_for(portraits, phase=True)
    embedded_w_tensor = tf.reshape(embedded_w, [portraits.shape[0], num_layers, latent_dim])
    
    latent_w = E.get_output_for(embedded_w_tensor, real_landmarks, phase=True)
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