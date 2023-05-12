# This script generates training dataset for pose estimation

import scipy.io as sio
import numpy as np
#from avi_r import AVIReader
#from preprocessing import bgsub
#from preprocessing import readVideo
from construct_model import f_x_to_model, add_noise
import matplotlib.pyplot as plt
#import time
import numpy.random as random
import os
import scipy.io as sio
import pdb
import cv2
import torch
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_folder', default="training_dataset", type=str, help='path to store training dataset')
parser.add_argument('-n','--n_samples',default=500000, type=int, help='number of training examples')
# User inputs
args = vars(parser.parse_args())
data_folder = args['data_folder']
n_samples = args['n_samples']

random.seed(10)

# Load the array of pose angles : generated_pose_all_2D_50k
# The angles are sampled from a probability distribution learnt from the  distribution of real poses (see manuscript)
theta_array = sio.loadmat('generated_pose_all_2D_50k.mat')
theta_array = theta_array['generated_pose']
if not os.path.exists(data_folder):
    os.mkdir(data_folder)
    os.mkdir(data_folder + '/images');
    os.mkdir(data_folder + '/coor_2d');
else:
    print('Warning: data_folder already exists. Files might be overwritten')

# Initiate parameter vector x
x = np.zeros((11, ))
for i in tqdm(range(0, n_samples)):
    if i % (int(n_samples / 5)) == 0:
        print('Finished ' + str(i) + ' of ' + str(n_samples), flush=True)
    # Initiate random co-ordinates of the larva in the field of view
    x[0] = 20 * (random.rand(1) - 0.5) + 100
    x[1] = 20 * (random.rand(1) - 0.5) + 100
    # Initate random orientation of the larva
    x[2] = random.rand(1) * 2 * np.pi
    x[3:11] = theta_array[i, :]
    # Sample a fish length to generate physical model image
    fishlen = (np.random.rand(1) - 0.5) * 30 + 70
    idxlen = np.floor((fishlen - 62) / 1.05) + 1
    seglen = 5.6 + idxlen * 0.1
    # Render physical model image and the corresponding annotation
    graymodel, pt = f_x_to_model(x.T, seglen, 1)
    graymodel = np.uint8(255 * (graymodel / np.max(graymodel)))
    graymodel = add_noise('gauss', graymodel, 0.001 * 100 * random.rand(1), 0.002 * 5 * random.rand(1))
    pt = tensor = torch.tensor(pt, dtype=torch.float32)
    # Save rendered image and annotation
    cv2.imwrite(data_folder + '/images/im_' + str(i).rjust(6, "0") + '.png', np.uint8(graymodel))
    torch.save(pt, data_folder + '/coor_2d/ann_' + str(i).rjust(6, "0") + '.pt')
