# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Appearance Cosine Similarity (CSIM)."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

import config
from metrics import metric_base
from training import dataset
from training import misc

from sklearn.metrics.pairwise import cosine_similarity

#----------------------------------------------------------------------------

class CSIM(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu
        self.facenet = tf.saved_model.load(config.FACENET_DB_DIR) # facenet.db
        self.facenet.print_layers()
        
    def run_image_manipulation(self, E, Gs, Inv, portraits, landmarks, num_gpus):
        out_split = []
        num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
        
        with tf.device("/cpu:0"):
            in_split_portraits = tf.split(portraits, num_or_size_splits=num_gpus, axis=0)
            in_split_landmarks = tf.split(landmarks, num_or_size_splits=num_gpus, axis=0)
        
        for gpu in range(num_gpus):
            with tf.device("/gpu:%d" % gpu):
                in_landmarks_gpu = in_split_landmarks[gpu]
                in_portraits_gpu = in_split_portraits[gpu]
                
                embedded_w = Inv.get_output_for(in_portraits_gpu, phase=True)
                embedded_w_tensor = tf.reshape(embedded_w, [portraits.shape[0], num_layers, latent_dim])
                latent_w = E.get_output_for(embedded_w_tensor, in_landmarks_gpu, phase=False)
                latent_wp = tf.reshape(latent_w, [portraits.shape[0], num_layers, latent_dim])
                fake_X_val = Gs.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
                out_split.append(fake_X_val)

        with tf.device("/cpu:0"):
            out_expr = tf.concat(out_split, axis=0)

        return out_expr
    
    def _evaluate(self, E, Gs, Inv, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        resolution = Gs.components.synthesis.output_shape[2]
        
        placeholder_portraits = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 3, resolution, resolution], name='placeholder_portraits')
        placeholder_landmarks = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 3, resolution, resolution], name='placeholder_landmarks')
        
        fake_X_val = self.run_image_manipulation(E, Gs, Inv, placeholder_portraits, placeholder_landmarks, num_gpus)
        
        csim_sum = 0.0
        
        for idx, data in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
            batch_portraits = data[:,0,:,:,:]
            batch_landmarks = np.roll(data[:,1,:,:,:], shift=1, axis=0)

            batch_portraits = misc.adjust_dynamic_range(batch_portraits.astype(np.float32), [0, 255], [-1., 1.])
            batch_landmarks = misc.adjust_dynamic_range(batch_landmarks.astype(np.float32), [0, 255], [-1., 1.])

            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)
            samples_manipulated = tflib.run(fake_X_val, feed_dict={placeholder_portraits: batch_portraits, placeholder_landmarks: batch_landmarks})

            embeddings_real = self.facenet.run(batch_portraits, num_gpus=num_gpus, assume_frozen=True)
            embeddings_fake = self.facenet.run(samples_manipulated, num_gpus=num_gpus, assume_frozen=True)
            
            for i in range(minibatch_size):
                #TODO calculate csim
                csim_sum += cosine_similarity([embeddings_real[i], embeddings_fake[i]])
                
            if end == self.num_images:
                break
        avg_csim = csim_sum/self.num_images
        
        self._report_result(np.real(avg_csim))
        
    #TODO include landmarks
    def _iterate_reals(self, minibatch_size):
        dataset_obj = dataset.load_dataset(data_dir=config.data_dir, **self._dataset_args)
        while True:
            images, _labels = dataset_obj.get_minibatch_np(minibatch_size)
            yield images
#----------------------------------------------------------------------------
