# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Landmark Hausdorff (LMHausdorff)."""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
import tensorflow as tf
import dnnlib.tflib as tflib
from landmark_extractor.landmark_extractor import FaceLandmarkExtractor

import config
from metrics import metric_base
from training import dataset
from training import misc

from tqdm import tqdm

#----------------------------------------------------------------------------

class LMHausdorff(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu
        self.landmark_extractor = FaceLandmarkExtractor()
        
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
    
    #extract subsets for each landmark type from the landmark image
    def split_landmarks(self, landmark_image=None):
        resolution = landmark_image.shape[0]

        chin = []
        eyebrows = []
        nose = []
        eyes = []
        mouth = []
        
        for i in range(resolution):
            for j in range(resolution):
                color_ij = np.rint(landmark_image[i][j])
                #black -> background
                if (color_ij[0] == 0) and (color_ij[1] == 0) and (color_ij[2] == 0):
                    continue
                #green -> chin
                elif (color_ij[0] == 0) and (color_ij[1] == 128) and (color_ij[2] == 0):
                    chin.append((i/resolution, j/resolution))
                #orange -> eyebrows
                elif (color_ij[0] == 255) and (color_ij[1] == 165) and (color_ij[2] == 0):
                    eyebrows.append((i/resolution, j/resolution))
                #blue -> nose
                elif (color_ij[0] == 0) and (color_ij[1] == 0) and (color_ij[2] == 255):
                    nose.append((i/resolution, j/resolution))
                #red -> eyes
                elif (color_ij[0] == 255) and (color_ij[1] == 0) and (color_ij[2] == 0):
                    eyes.append((i/resolution, j/resolution))
                #pink -> mouth
                elif (color_ij[0] == 255) and (color_ij[1] == 192) and (color_ij[2] == 203):
                    mouth.append((i/resolution, j/resolution))

        return np.asarray([chin, eyebrows, nose, eyes, mouth])
    
    def calculate_landmark_hausdorff(self, lm_img1, lm_img2):
        #get landmark subsets
        set1 = self.split_landmarks(lm_img1)
        set2 = self.split_landmarks(lm_img2)

        # Calculate Hausdorff Distance.
        hd_dist = 0.0
        for i in range(5):
          hd_dist += max(directed_hausdorff(set1[i], set2[i])[0], directed_hausdorff(set2[i], set1[i])[0])
            
        return hd_dist
    
    def _evaluate(self, Gs, E, Inv, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        resolution = Gs.components.synthesis.output_shape[2]
        
        placeholder_portraits = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 3, resolution, resolution], name='placeholder_portraits')
        placeholder_landmarks = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 3, resolution, resolution], name='placeholder_landmarks')
        placeholder_keypoints = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 136], name='placeholder_landmarks')
        
        fake_X_val = self.run_image_manipulation(E, Gs, Inv, placeholder_portraits, placeholder_landmarks, placeholder_keypoints, num_gpus)
        
        hd_sum = 0.0
        failed_counter = 0
        
        for idx, data in tqdm(enumerate(self._iterate_reals(minibatch_size=minibatch_size))):
            image_data = data[0]
            batch_portraits = image_data[:,0,:,:,:]
            batch_landmarks = np.roll(image_data[:,1,:,:,:], shift=1, axis=0)
            
            keypoints = np.roll(data[1], shift=1, axis=0)

            batch_portraits = misc.adjust_dynamic_range(batch_portraits.astype(np.float32), [0, 255], [-1., 1.])
            batch_landmarks = misc.adjust_dynamic_range(batch_landmarks.astype(np.float32), [0, 255], [-1., 1.])

            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)
            samples_manipulated = tflib.run(fake_X_val, feed_dict={placeholder_portraits: batch_portraits, placeholder_landmarks: batch_landmarks})
            
            samples_manipulated = misc.adjust_dynamic_range(samples_manipulated.astype(np.float32), [-1., 1.], [0, 255])
            samples_manipulated = np.transpose(samples_manipulated, [0, 2, 3, 1])

            batch_landmarks = misc.adjust_dynamic_range(batch_landmarks.astype(np.float32), [-1., 1.], [0, 255])

            for i in range(minibatch_size):
                try:
                  ground_truth_lm = batch_landmarks[i]
                  generated_lm, _ = self.landmark_extractor.generate_landmark_image(source_path_or_image=samples_manipulated[i], resolution=resolution)
                  generated_lm = generated_lm.cpu().detach().numpy()
                  ground_truth_lm = np.transpose(ground_truth_lm, [1, 2, 0])
                  hd_sum += self.calculate_landmark_hausdorff(ground_truth_lm, generated_lm)
                
                except Exception as e:
                  print('Error: Landmark couldnt be extracted from generated output. Skipping the sample')
                  failed_counter += 1
                  continue
                
            if end == self.num_images:
                break
        avg_hd_dist = hd_sum/(self.num_images - failed_counter)
        
        self._report_result(np.real(avg_hd_dist), suffix='/Average Landmark Hausdorff Distance')
        self._report_result(failed_counter, suffix='/Number of failed landmark extractions')
        
    #TODO include landmarks
    def _iterate_reals(self, minibatch_size):
        dataset_obj = dataset.load_dataset(data_dir=config.data_dir, **self._dataset_args)
        while True:
            images, _labels = dataset_obj.get_minibatch_np(minibatch_size)
            yield images
#----------------------------------------------------------------------------
