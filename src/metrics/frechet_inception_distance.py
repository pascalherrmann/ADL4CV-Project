# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from tqdm import tqdm

import config
from metrics import metric_base
from training import misc

from utils.visualizer import save_image
from utils.visualizer import adjust_pixel_range
from utils.visualizer import fuse_images


def announce(msg):
    print("\n"*3 + "="*60 + "\n{}\n".format(msg) + "="*60)

#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, E, Inv, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl(config.INCEPTION_PICKLE_DIR) # inception_v3_features.pkl
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        announce("Evaluating Reals")
        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)
            print("loaded real mu, sigma from cache.")
        else:
            progress = 0
            for idx, batch_stacks in tqdm(enumerate(self._iterate_reals(minibatch_size=minibatch_size)), position=0, leave=True):
                progress += batch_stacks.shape[0]
                images = batch_stacks[:,0,:,:,:]
                landmarks = batch_stacks[:,1,:,:,:]

                # compute inception on full images!!!
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)


                # visualization
                images = images.astype(np.float32) / 255 * 2.0 - 1.0
                landmarks = landmarks.astype(np.float32) / 255 * 2.0 - 1.0

                if idx <= 10:
                    debug_img = np.concatenate([
                        images, # original landmarks
                        landmarks # original portraits,
                    ], axis=0)
                    debug_img = adjust_pixel_range(debug_img)
                    debug_img = fuse_images(debug_img, row=2, col=minibatch_size)
                    save_image("data_iter_{}08d.png".format(idx), debug_img)
                if end == self.num_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            misc.save_pkl((mu_real, sigma_real), cache_file)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
        
        announce("Evaluating Generator.")
        # Construct TensorFlow graph.
        result_expr = []
        print("Construct TensorFlow graph.")
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                images = Gs_clone.get_output_for(latents, None, is_validation=True, randomize_noise=True)
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception_clone.get_output_for(images))

        # Calculate statistics for fakes.
        print("Calculate statistics for fakes.")
        for begin in tqdm(range(0, self.num_images, minibatch_size), position=0, leave=True):
            end = min(begin + minibatch_size, self.num_images)
            #print("result_expr", len(result_expr)) # result_expr is a list!!!
            # results_expr[0].shape = (8, 2048) -> hat nur ein element.
            # weil: eigentlich würde man halt hier die GPUs zusammen konkattenieren.

            res_expr, fakes = tflib.run([result_expr, images])
            activations[begin:end] = np.concatenate(res_expr, axis=0)[:end-begin]

            if begin < 20:
                fakes = fakes.astype(np.float32) / 255 * 2.0 - 1.0
                debug_img = np.concatenate([
                    fakes
                ], axis=0)
                debug_img = adjust_pixel_range(debug_img)
                debug_img = fuse_images(debug_img, row=3, col=minibatch_size)
                save_image("fid_generator_iter_{}08d.png".format(end), debug_img)


        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        #print("mu_fake={}, sigma_fake={}".format(mu_fake, sigma_fake))
        
        # Calculate FID.
        print("Calculate FID (generator).")
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist), suffix="StyleGAN Generator Only")
        print("Distance StyleGAN", dist)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

        announce("Now evaluating encoder (appearnace)")
        print("building custom encoder graph!")
        with tf.variable_scope('fakeddddoptimizer'):

            # Build graph.
            BATCH_SIZE = self.minibatch_per_gpu
            input_shape = Inv.input_shape
            input_shape[0] = BATCH_SIZE
            latent_shape = Gs.components.synthesis.input_shape
            latent_shape[0] = BATCH_SIZE

            x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
            x_lm = tf.placeholder(tf.float32, shape=input_shape, name='some_landmark')

            if self.model_type == "rignet":
                w_enc_1 = Inv.get_output_for(x, phase=False)
                wp_enc_1 = tf.reshape(w_enc_1, latent_shape)
                w_enc = E.get_output_for(wp_enc_1, x_lm, phase=False)
            else:
                w_enc = E.get_output_for(x, x_lm, phase=False)

            wp_enc = tf.reshape(w_enc, latent_shape)

            manipulated_images = Gs.components.synthesis.get_output_for(wp_enc, randomize_noise=False)
            manipulated_images = tflib.convert_images_to_uint8(manipulated_images)
            inception_codes = inception_clone.get_output_for(manipulated_images) # shape (8, 2048)

        for idx, batch_stacks in tqdm(enumerate(self._iterate_reals(minibatch_size=minibatch_size)), position=0, leave=True):

            images = batch_stacks[:,0,:,:,:]    # shape (8, 3, 128, 128)
            landmarks = batch_stacks[:,1,:,:,:] # shape (8, 3, 128, 128)
            images = images.astype(np.float32) / 255 * 2.0 - 1.0
            landmarks = landmarks.astype(np.float32) / 255 * 2.0 - 1.0

            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images) # begin: 0; end: 8

            activations[begin:end], manip  = tflib.run([inception_codes, manipulated_images], feed_dict={x:images, x_lm:landmarks})
            # acivations: (5000, 2048)



            if idx < 10:
                print("saving img")
                manip = manip.astype(np.float32) / 255 * 2.0 - 1.0
                debug_img = np.concatenate([
                    images, # original landmarks
                    landmarks, # original portraits,
                    manip
                ], axis=0)
                debug_img = adjust_pixel_range(debug_img)
                debug_img = fuse_images(debug_img, row=3, col=minibatch_size)
                save_image("fid_iter_{}08d.png".format(idx), debug_img)


            if end == self.num_images:
                break

        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)
        #print("enc_mu_fake={}, enc_sigma_fake={}".format(mu_fake, sigma_fake))


        # Calculate FID.
        print("Calculate FID for encoded samples")
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist), suffix="Our Face-Landmark-Encoder (Apperance)")
        print("distance OUR FACE-LANDMARK-ENCODER", dist)


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

        announce("Now evaluating encoder. (POSE)")
        print("building custom encoder graph!")
        with tf.variable_scope('fakeddddoptimizer'):

            # Build graph.
            BATCH_SIZE = self.minibatch_per_gpu
            input_shape = Inv.input_shape
            input_shape[0] = BATCH_SIZE
            latent_shape = Gs.components.synthesis.input_shape
            latent_shape[0] = BATCH_SIZE

            x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
            x_lm = tf.placeholder(tf.float32, shape=input_shape, name='some_landmark')
            x_kp = tf.placeholder(tf.float32, shape=[self.minibatch_per_gpu, 136], name='some_keypoints')

            if self.model_type == "rignet":
                w_enc_1 = Inv.get_output_for(x, phase=False)
                wp_enc_1 = tf.reshape(w_enc_1, latent_shape)
                w_enc = E.get_output_for(wp_enc_1, x_lm, phase=False)
            elif self.model_type == "keypoints":
                w_enc_1 = Inv.get_output_for(x, phase=False)
                wp_enc_1 = tf.reshape(w_enc_1, latent_shape)
                w_enc = E.get_output_for(wp_enc_1, x_kp, phase=False)
            else:
                w_enc = E.get_output_for(x, x_lm, phase=False)

            wp_enc = tf.reshape(w_enc, latent_shape)

            manipulated_images = Gs.components.synthesis.get_output_for(wp_enc, randomize_noise=False)
            manipulated_images = tflib.convert_images_to_uint8(manipulated_images)
            inception_codes = inception_clone.get_output_for(manipulated_images) # shape (8, 2048)

        for idx, data in tqdm(enumerate(self._iterate_reals(minibatch_size=minibatch_size)), position=0, leave=True):

            image_data = data[0]
            images = image_data[:,0,:,:,:]
            landmarks = np.roll(image_data[:,1,:,:,:], shift=1, axis=0)
            
            keypoints = np.roll(data[1], shift=1, axis=0)

            images = images.astype(np.float32) / 255 * 2.0 - 1.0
            landmarks = landmarks.astype(np.float32) / 255 * 2.0 - 1.0

            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images) # begin: 0; end: 8

            activations[begin:end], manip  = tflib.run([inception_codes, manipulated_images], feed_dict={x:images, x_lm:landmarks, x_kp:keypoints})
            # acivations: (5000, 2048)



            if idx < 10:
                print("saving img")
                manip = manip.astype(np.float32) / 255 * 2.0 - 1.0
                debug_img = np.concatenate([
                    images, # original landmarks
                    landmarks, # original portraits,
                    manip
                ], axis=0)
                debug_img = adjust_pixel_range(debug_img)
                debug_img = fuse_images(debug_img, row=3, col=minibatch_size)
                save_image("fid_iter_POSE_{}08d.png".format(idx), debug_img)


            if end == self.num_images:
                break

        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)
        #print("enc_mu_fake={}, enc_sigma_fake={}".format(mu_fake, sigma_fake))


        # Calculate FID.
        print("Calculate FID for encoded samples (POSE)")
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist), suffix="Our_Face_Landmark_Encoder (Pose)")
        print("distance OUR FACE-LANDMARK-ENCODER (POSE)", dist)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

        announce("Now in domain inversion only encoder.")
        print("building custom in domain inversion graph!")
        with tf.variable_scope('fakedddwdoptimizer'):

            # Build graph.
            BATCH_SIZE = self.minibatch_per_gpu
            input_shape = Inv.input_shape
            input_shape[0] = BATCH_SIZE
            latent_shape = Gs.components.synthesis.input_shape
            latent_shape[0] = BATCH_SIZE

            x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')

            w_enc_1 = Inv.get_output_for(x, phase=False)
            wp_enc_1 = tf.reshape(w_enc_1, latent_shape)

            manipulated_images = Gs.components.synthesis.get_output_for(wp_enc_1, randomize_noise=False)
            manipulated_images = tflib.convert_images_to_uint8(manipulated_images)
            inception_codes = inception_clone.get_output_for(manipulated_images)

        for idx, batch_stacks in tqdm(enumerate(self._iterate_reals(minibatch_size=minibatch_size)), position=0, leave=True):

            images = batch_stacks[:,0,:,:,:]
            landmarks = batch_stacks[:,1,:,:,:]
            images = images.astype(np.float32) / 255 * 2.0 - 1.0
            landmarks = landmarks.astype(np.float32) / 255 * 2.0 - 1.0

            #print("landmarks", landmarks.shape)# (8, 3, 128, 128)
            #print("images", images.shape) # (8, 3, 128, 128)
            #print("inception_codes", inception_codes.shape) # (8, 2048)
            #print("activations", activations.shape) # (5000, 2048)
            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)
            #print("b,e", begin, end) # 0, 8; ...

            activations[begin:end]  = tflib.run(inception_codes, feed_dict={x:images})

            if end == self.num_images:
                break

        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)
        #print("enc_mu_fake={}, enc_sigma_fake={}".format(mu_fake, sigma_fake))


        # Calculate FID.
        print("Calculate FID for IN-DOMAIN-GAN-INVERSION")
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist), suffix="_In-Domain-Inversion_Only")
        print("distance IN-DOMAIN-GAN-INVERSION:", dist)
