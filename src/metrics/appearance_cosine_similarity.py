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

import scipy
from tensorflow.python.platform import gfile

#----------------------------------------------------------------------------

class CSIM(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

        with gfile.FastGFile(config.FACENET_PB_DIR,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=None, name='')
        self.facenet_graph = tf.get_default_graph()

    def get_facenet_embeddings(self, images):
        # Get input and output tensors
        images_placeholder = self.facenet_graph.get_tensor_by_name("input:0")
        embeddings = self.facenet_graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder =self.facenet_graph.get_tensor_by_name("phase_train:0")
        
        # Run forward pass to calculate embeddings
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        with tf.Session(graph=self.facenet_graph) as sess:
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
      
        return emb_array

def run_image_manipulation(self, E, Gs, Inv, portraits, landmarks, keypoints, num_gpus):
        out_split = []
        num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
        
        with tf.device("/cpu:0"):
            in_split_portraits = tf.split(portraits, num_or_size_splits=num_gpus, axis=0)
            in_split_landmarks = tf.split(landmarks, num_or_size_splits=num_gpus, axis=0)
            in_split_keypoints = tf.split(keypoints, num_or_size_splits=num_gpus, axis=0)
        
        for gpu in range(num_gpus):
            with tf.device("/gpu:%d" % gpu):
                in_landmarks_gpu = in_split_landmarks[gpu]
                in_portraits_gpu = in_split_portraits[gpu]
                in_keypoints_gpu = in_split_keypoints[gpu]
                
                if self.model_type == "rignet":
                    embedded_w = Inv.get_output_for(in_portraits_gpu, phase=True)
                    embedded_w_tensor = tf.reshape(embedded_w, [portraits.shape[0], num_layers, latent_dim])
                    latent_w = E.get_output_for(embedded_w_tensor, in_landmarks_gpu, phase=False)
                elif self.model_type == 'keypoints':
                    embedded_w = Inv.get_output_for(in_portraits_gpu, phase=True)
                    embedded_w_tensor = tf.reshape(embedded_w, [portraits.shape[0], num_layers, latent_dim])
                    latent_w = E.get_output_for(embedded_w_tensor, in_keypoints_gpu, phase=False)
                else:
                    latent_w = E.get_output_for(in_portraits_gpu, in_landmarks_gpu, phase=False)

                latent_wp = tf.reshape(latent_w, [portraits.shape[0], num_layers, latent_dim])
                fake_X_val = Gs.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
                out_split.append(fake_X_val)

        with tf.device("/cpu:0"):
            out_expr = tf.concat(out_split, axis=0)

        return out_expr
    
    def _evaluate(self, Gs, E, Inv, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        resolution = Gs.components.synthesis.output_shape[2]
        
        placeholder_portraits = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 3, resolution, resolution], name='placeholder_portraits')
        placeholder_landmarks = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 3, resolution, resolution], name='placeholder_landmarks')
        placeholder_keypoints = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 136], name='placeholder_landmarks')
        
        fake_X_val = self.run_image_manipulation(E, Gs, Inv, placeholder_portraits, placeholder_landmarks, placeholder_keypoints, num_gpus)
        
        csim_sum = 0.0
        
        for idx, data in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
            image_data = data[0]
            batch_portraits = image_data[:,0,:,:,:]
            batch_landmarks = np.roll(image_data[:,1,:,:,:], shift=1, axis=0)
            
            keypoints = np.roll(data[1], shift=1, axis=0)

            batch_portraits = misc.adjust_dynamic_range(batch_portraits.astype(np.float32), [0, 255], [-1., 1.])
            batch_landmarks = misc.adjust_dynamic_range(batch_landmarks.astype(np.float32), [0, 255], [-1., 1.])

            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)
            samples_manipulated = tflib.run(fake_X_val, feed_dict={placeholder_portraits: batch_portraits, placeholder_landmarks: batch_landmarks, placeholder_keypoints: keypoints})
            
            samples_manipulated = np.transpose(samples_manipulated, [0, 2, 3, 1])
            samples_manipulated = np.pad(samples_manipulated, ((0, 0), (11, 11), (11, 11), (0, 0)), mode='constant')

            batch_portraits = np.transpose(batch_portraits, [0, 2, 3, 1])
            batch_portraits = np.pad(batch_portraits, ((0, 0), (11, 11), (11, 11), (0, 0)), mode='constant')

            embeddings_real = self.get_facenet_embeddings(batch_portraits)
            embeddings_fake = self.get_facenet_embeddings(samples_manipulated)
            
            for i in range(minibatch_size):
                csim_sum += 1 - scipy.spatial.distance.cosine(embeddings_real[i], embeddings_fake[i])
            
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
