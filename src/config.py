# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Global configuration."""

#----------------------------------------------------------------------------
# StyleGAN-Configuration. No need to change.

result_dir = 'results'
data_dir = 'datasets'
cache_dir = 'cache'
run_dir_ignore = ['results', 'datasets', 'cache']

#----------------------------------------------------------------------------

# =======================
# Begin of Configuration
# =======================
RESOLUTION = 128
DESC = ""

DATA_DIR = "/content/gdrive/My Drive/Public/tensorboards_shared/split_dataset/train"
TEST_DATA_DIR = "/content/gdrive/My Drive/Public/tensorboards_shared/split_dataset/test/dataset_tfrecords0-r07.tfrecords"
PICKLE_DIR = "/content/gdrive/My Drive/Public/tensorboards_shared/Training_Decoder_TF/00000-sgan-ffhq128-1gpu/network-snapshot-011489.pkl"

ENCODER_PICKLE_DIR = None
# Encoder
GDRIVE_PATH = "/content/gdrive/My Drive/Public/tensorboards_shared"

# only for decoder
TIME = 0
KIMG = 0

# =======================
# End of Configuration
# =======================
import os

def getBranch():
    branch = os.popen('git rev-parse --abbrev-ref HEAD').read()
    branch_without_line_break = branch[:-1]
    return branch_without_line_break

def getGdrivePath():
    return os.path.join(GDRIVE_PATH, getBranch())

def checkBranch():
    branch = getBranch()
    return branch[:4] == "run/"

def makeLogDir():
    path = getGdrivePath()
    path_imgs = os.path.join(path, "images")
    path_snapshots = os.path.join(path, "snapshots")
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_imgs, exist_ok=True)
    os.makedirs(path_snapshots, exist_ok=True)
    print("Created Log-Directory {}".format(path))
