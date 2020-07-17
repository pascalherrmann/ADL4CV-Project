import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from tqdm import tqdm

from utils.visualizer import save_image, load_image, resize_image, adjust_pixel_range

from metrics import metric_base
from training import misc

import config
from metrics.util import convert_pickle_path_to_name



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def get_test_images_paths(dir):
    img_paths = []
    for file in os.listdir(dir):
        if file.endswith(".png"):
            img_paths.append(os.path.join(dir, file))
    return sorted(img_paths)

def load_test_images(dir, image_size = 128):
    img_paths = get_test_images_paths(dir)
    imgs = []

    for i, image_path in enumerate(img_paths):
        IMG = resize_image(load_image(image_path), (image_size, image_size))
        imgs.append(IMG)

    return imgs

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


class Gallery(metric_base.MetricBase):
    def __init__(self, img_folder, lm_folder, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.img_folder = img_folder
        self.lm_folder = lm_folder
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, E, Inv, num_gpus):

        loaded_imgs = load_test_images(self.img_folder)
        loaded_landmarks = load_test_images(self.lm_folder)
        minibatch_size = num_gpus * self.minibatch_per_gpu







        ##
        # Set up graph
        ##


        # Get input size.
        image_size = Inv.input_shape[2]
        assert image_size == Inv.input_shape[3]
        input_shape = Inv.input_shape
        input_shape[0] = self.minibatch_per_gpu
        latent_shape = Gs.components.synthesis.input_shape
        latent_shape[0] = self.minibatch_per_gpu

        # Build graph.
        print(f'Building graph.')
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




        def get_landmark_row(lm, img_list):
            displayed_imgs = [lm]

            # the np variables we'll build to feed to graph!
            batch_images = np.zeros(input_shape, np.uint8)
            batch_lms = np.zeros(input_shape, np.uint8)

            for img_idx in tqdm(range(0, len(img_list), self.minibatch_per_gpu), leave=False):
                batch = img_list[img_idx:img_idx + self.minibatch_per_gpu]
                for i, image in enumerate(batch):
                    batch_images[i] = np.transpose(image, [2, 0, 1])
                    batch_lms[i] = np.transpose(lm, [2, 0, 1])

                inputs = batch_images.astype(np.float32) / 255 * 2.0 - 1.0
                inputs_lm = batch_lms.astype(np.float32) / 255 * 2.0 - 1.0

                # Run encoder.
                outputs = tflib.run(manipulated_images, {x: inputs, x_lm: inputs_lm})
                outputs = adjust_pixel_range(outputs, min_val=0, max_val=255) # 16 x 128 x 128 x 3
                for i, _ in enumerate(batch):
                    displayed_imgs.append(outputs[i])

            return displayed_imgs


        header = [(np.ones([image_size,image_size, 3])*255).astype(np.uint8)] # start with empty square for landmark slot
        header.extend(loaded_imgs)

        results = [header]
        print("rsultss", len(results))
        print("resuls[0]", len(results[0]))
        for sample_landmark in loaded_landmarks:
            current_result = get_landmark_row(sample_landmark, loaded_imgs)
            results.append(current_result)




        def get_gallery_array(img_lists):
            stacks = []
            for img_list in img_lists:
                stack = np.concatenate(img_list, axis=1)
                stacks.append(stack)

            total_stack = np.concatenate(stacks, axis=0)
            return total_stack


        arr = get_gallery_array(results)
        path = os.path.join(config.EVALUATION_DIR, self.name + "_" + convert_pickle_path_to_name(self._network_pkl) + ".png")
        save_image(path, arr)


        def display_image_in_actual_size(img_arr, title):

            dpi = 80
            height, width, depth = img_arr.shape

            # What size does the figure need to be in inches to fit the image?
            figsize = width / float(dpi), height / float(dpi)

            # Create a figure of the right size with one axes that takes up the full figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])

            # Hide spines, ticks, etc.
            ax.axis('off')

            # Display the image.
            ax.imshow(img_arr, cmap='gray')
            fig.suptitle("Model-Path: \"{}\"".format(title), fontsize=12, y=0)
            plt.show()


        display_image_in_actual_size(arr, self._network_pkl)

#----------------------------------------------------------------------------
