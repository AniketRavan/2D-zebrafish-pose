import scipy.io as sio
import numpy as np
#from avi_r import AVIReader
#from preprocessing import bgsub
#from preprocessing import readVideo
from construct_model import f_x_to_model, add_noise, generate_mask
import matplotlib.pyplot as plt
#import time
import numpy.random as random
import os
import scipy.io as sio
import pdb
import cv2
import torch
import argparse

random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_folder', default="../../../training_dataset", type=str, help='path to store training dataset')
parser.add_argument('-n','--n_samples',default=500000, type=int, help='number of training examples')
args = vars(parser.parse_args())

data_folder = args['data_folder']
n_samples = args['n_samples']

theta_array = sio.loadmat('generated_pose_all_2D_50k.mat');
theta_array = theta_array['generated_pose']

if not os.path.exists(data_folder):
    os.mkdir(data_folder)
    os.mkdir(data_folder + '/images');
    os.mkdir(data_folder + '/masks');
    os.mkdir(data_folder + '/coor_2d');
else:
    print('Target folder already exists. This may cause files might be overwritten. Consider renaming target folder')

def generate_mask(img):
    _, bw = cv2.threshold(np.uint8(img), 0, 255, cv2.THRESH_BINARY)
    # Choose dilation kernel whose size is randomly sampled from [3, 5, 7]
    kernel_size = random.randint(1,4) * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype = np.uint8)
    bw_dilated = cv2.dilate(bw, kernel, iterations = 1)
    return bw_dilated 

x = np.zeros((11, ))
for i in range(0, n_samples):
    if i % 1000 == 0:
        print('Finished ' + str(i) + ' of 10000', flush=True)
    x[0] = 20 * (random.rand(1) - 0.5) + 100
    x[1] = 20 * (random.rand(1) - 0.5) + 100
    x[2] = random.rand(1) * 2 * np.pi
    x[3:11] = theta_array[i, :]
    fishlen = (np.random.rand(1) - 0.5) * 30 + 70
    idxlen = np.floor((fishlen - 62) / 1.05) + 1
    seglen = 5.6 + idxlen * 0.1
    graymodel, pt = f_x_to_model(x.T, seglen, 1)
    graymodel = np.uint8(255 * (graymodel / np.max(graymodel)))
    mask = generate_mask(graymodel)
    graymodel = add_noise('gauss', graymodel, 0.001 * 100 * random.rand(1), 0.002 * 5 * random.rand(1))
    #graymodel[mask == 0] = 0
    pt = torch.tensor(pt, dtype=torch.float32)
    cv2.imwrite(data_folder + '/images/im_' + str(i).rjust(6, "0") + '.png', np.uint8(graymodel))
    cv2.imwrite(data_folder + '/masks/bw_' + str(i).rjust(6, "0") + '.png', np.uint8(mask))
    torch.save(pt, data_folder + '/coor_2d/ann_' + str(i).rjust(6, "0") + '.pt')
