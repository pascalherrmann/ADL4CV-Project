# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:33:20 2020

@author: Tobias Zengerle
"""

import os
import torch
from torch.utils.data import Dataset
from skimage import io

class PortraitLandmarkDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, portrait_root_dir, landmark_root_dir, transform=None):
        """
        Args:
            portrait_root_dir (string): Directory with all the portrait images
            portrait_root_dir (string): Directory with all the landmark images.
        """
        self.portrait_root_dir = portrait_root_dir
        self.landmark_root_dir = landmark_root_dir
        self.transform = transform

    def __len__(self):
        #TODO automatically set
        return 1024#len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        folder_name = str(int(int(idx * 0.001) / 0.001)).zfill(5)
        portrait_img_name = os.path.join(self.portrait_root_dir, folder_name)
        portrait_img_name = os.path.join(portrait_img_name, str(idx).zfill(5) + '.png')
        portrait_img = io.imread(portrait_img_name)
        
        landmark_img_name = os.path.join(self.landmark_root_dir, folder_name)
        landmark_img_name = os.path.join(landmark_img_name, str(idx).zfill(5) + '.png')
        landmark_img = io.imread(landmark_img_name)
        
        # apply the transforms on the images
        if self.transform is not None:
            portrait_img = self.transform(portrait_img)
            landmark_img = self.transform(landmark_img)

        # ignore the alpha channel
        # in the images if it exists
        if portrait_img.shape[2] >= 4:
            portrait_img = portrait_img[:, :, :3]
            
        if landmark_img.shape[2] >= 4:
            landmark_img = landmark_img[:, :, :3]
        
        sample = {'portrait_image': portrait_img, 'landmark_image': landmark_img}

        return sample
    
class PortraitLandmarkDatasetPriorSetup(Dataset):
    """ pyTorch Dataset wrapper for folder distributed dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: portrait_files => list of paths of portrait files
        :return: landmark_files => list of paths of landmark files
        """

        #it is assumed that the directories for the portrait and landmark data are identically structured
        dir_names = os.listdir(self.portrait_root_dir)
        portrait_files = []  # initialize to empty list
        landmark_files = []  # initialize to empty list

        for dir_name in dir_names:
            try:
                portrait_file_path = os.path.join(self.portrait_root_dir, dir_name)
                landmark_file_path = os.path.join(self.landmark_root_dir, dir_name)
                
                portrait_file_names = os.listdir(portrait_file_path)
                #landmark_file_names = os.listdir(landmark_file_path)
                for file_name in portrait_file_names:
                    possible_portrait_file = os.path.join(portrait_file_path, file_name)
                    possible_landmark_file = os.path.join(landmark_file_path, file_name)
                    if os.path.isfile(possible_portrait_file) and os.path.isfile(possible_landmark_file):
                        portrait_files.append(possible_portrait_file)
                        landmark_files.append(possible_landmark_file)
            except:
                print('Error: Problem with loading an image file. Dataset setup proceeds...')
                continue

        # return the files list
        return portrait_files, landmark_files

    def __init__(self, portrait_root_dir, landmark_root_dir, transform=None):
        """
        constructor for the class
        :param portrait_root_dir: path to the directory containing the portrait data
        :param landmark_root_dir: path to the directory containing the landmark data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.portrait_root_dir = portrait_root_dir
        self.landmark_root_dir = landmark_root_dir
        self.transform = transform

        # setup the files for reading
        self.portrait_files, self.landmark_files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.portrait_files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        #from PIL import Image

        # read the portrait image:
        portrait_img_name = self.portrait_files[idx]
#        if portrait_img_name[-4:] == ".npy":
#            portrait_img = np.load(portrait_img_name)
#            portrait_img = Image.fromarray(portrait_img.squeeze(0).transpose(1, 2, 0))
#        else:
        portrait_img = io.imread(portrait_img_name)
            
        # read the landmark image:
        landmark_img_name = self.landmark_files[idx]
#        if landmark_img_name[-4:] == ".npy":
#            landmark_img = np.load(landmark_img_name)
#            landmark_img = Image.fromarray(landmark_img.squeeze(0).transpose(1, 2, 0))
#        else:
        landmark_img = io.imread(landmark_img_name)

        # apply the transforms on the images
        if self.transform is not None:
            portrait_img = self.transform(portrait_img)
            landmark_img = self.transform(landmark_img)

        # ignore the alpha channel
        # in the images if it exists
        if portrait_img.shape[2] >= 4:
            portrait_img = portrait_img[:, :, :3]
            
        if landmark_img.shape[2] >= 4:
            landmark_img = landmark_img[:, :, :3]

        # return the image tuple
        sample = {'portrait_image': portrait_img, 'landmark_image': landmark_img}
        return sample