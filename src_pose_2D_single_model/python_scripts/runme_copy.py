import scipy.io as sio
import numpy as np
#from avi_r import AVIReader
#from preprocessing import bgsub
#from preprocessing import readVideo
from construct_model import f_x_to_model
import matplotlib.pyplot as plt
#import time
import numpy.random as random
import os
import scipy.io as sio
import pdb
import cv2
import torch

random.seed(10)

theta_array = sio.loadmat('generated_pose_all_2D_50k.mat');
theta_array = theta_array['generated_pose']
data_folder = 'training_data_2D_230602';
#os.mkdir(data_folder)
#os.mkdir(data_folder + '/images');
#os.mkdir(data_folder + '/coor_2d');

def add_noise(noise_typ,image,mean,var):
   if noise_typ == "gauss":
      row,col= image.shape
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,1)) * 255
      gauss[gauss < 0] = 0
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      noisy[noisy > 255] = 255
      return noisy


x = np.zeros((11, ))
for i in range(0, 500000):
    if i % 1000 == 0:
        print('Finished ' + str(i) + 'of 500000', flush=True)
    x[0] = 20 * (random.rand(1) - 0.5) + 100
    x[1] = 20 * (random.rand(1) - 0.5) + 100
    x[2] = random.rand(1) * 2 * np.pi
    x[3:11] = theta_array[i, :]
    fishlen = (np.random.rand(1) - 0.5) * 30 + 70
    idxlen = np.floor((fishlen - 62) / 1.05) + 1
    seglen = 5.6 + idxlen * 0.1
    graymodel, pt = f_x_to_model(x.T, seglen, 1)
    graymodel = np.uint8(255 * (graymodel / np.max(graymodel)))
    graymodel = add_noise('gauss', graymodel, 0.001 * 100 * random.rand(1), 0.002 * 5 * random.rand(1))
    pt = tensor = torch.tensor(pt, dtype=torch.float32)
    cv2.imwrite(data_folder + '/images/im_' + str(i).rjust(6, "0") + '.png', np.uint8(graymodel))
    torch.save(pt, data_folder + '/coor_2d/ann_' + str(i).rjust(6, "0") + '.pt')
