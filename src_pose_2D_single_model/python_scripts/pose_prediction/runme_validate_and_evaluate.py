# This script generates pose predictions using a trained network model and evaluates the prediction using a correlation coefficient score
# The correlation coefficient scores are saved in a user-defined directory

import sys
sys.path.append("../src_pose_2D_single_model/")
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from CustomDataset_images_only import CustomImageDataset
from ResNet_Blocks_3D_four_blocks import resnet18
import time
from multiprocessing import Pool
import os
import scipy.io as sio
from scipy.optimize import least_squares
import numpy as np
import time
import pdb
from evaluation_functions import evaluate_prediction
from construct_model import f_x_to_model_evaluation

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=1, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="validations", type=str, help='path to store output images and plots')
parser.add_argument('-m','--model',default="resnet_pose_230506_best_python_four_blocks.pt", type=str, help='path to model file')
parser.add_argument('-i','--image_folder', default='../../validation_data_for_github',type=str,help='path to real images for network validation')
args = vars(parser.parse_args())

imageSizeX = 101
imageSizeY = 101

epochs = args['epochs']
output_dir = args['output_dir']
im_folder = args['image_folder']
model_path = args['model']
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(1, 12, activation='leaky_relu').to(device)
n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
    model.load_state_dict(torch.load(model_path))
    nworkers = n_cuda*16
    pftch_factor = 2
else:
    print('Cuda is not available')
    nworkers = 1
    pftch_factor = None
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
batch_size = 512

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

class AddGaussianNoise(object):
    def __init__(self, mean=0.005, std=0.0015):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class padding:
    def __call__(self, image):
        w, h = image.size
        w_buffer = 101 - w
        w_left = int(w_buffer/2)
        w_right = w_buffer - w_left
        w_buffer = 101 - h
        w_top = int(w_buffer/2)
        w_bottom = w_buffer - w_top
        padding = (w_left, w_top, w_right, w_bottom)
        pad_transform = transforms.Pad(padding)
        padded_image = pad_transform(image)
        return padded_image

transform = transforms.Compose([padding(), transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])
#im_folder = '../validation_data_fs_2D_' + date + '_subset/images_real/'

im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

val_data = CustomImageDataset(im_files_add, transform=transform)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='none')


def save_images(pose_recon, im, counter, output_dir):
    for i in range(pose_recon.shape[0]):
        _,axs = plt.subplots(nrows=3, ncols=2)
        axs[0,1].imshow(im[i,:,:].cpu(), cmap='gray')
        axs[0,1].scatter(pose_recon[i,0,:], pose_recon[i,1,:], s=0.07, c='green', alpha=0.6)
        axs[0,1].grid(False)
        axs[0,1].set_axis_off()
        axs[0,0].imshow(im[i,:,:].cpu(), cmap='gray')
        axs[0,0].scatter(pose_data[i,0,:], pose_data[i,1,:], s=0.07, c='red', alpha=0.6)
        axs[0,0].grid(False)
        axs[0,0].set_axis_off()

        plt.savefig(output_dir + '/im_' + str(counter) + '.svg')
        plt.close()
        counter = counter + 1

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    pose_loss_array = []
    counter = 0
    reconstructed_pose = []
    idx = 0
    corr_coef = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            #for i, data in enumerate(dataloader):
            for p in range(0,1): 
                im_cpu, filename = data
                im = im_cpu.to(device)
                pose_recon = model(im)
                pose_recon = pose_recon.cpu()
                reconstructed_pose.append(pose_recon)
                for batch_sample in range(0, im_cpu.shape[0]):
                    corr_coef.append(evaluate_prediction(np.squeeze(im_cpu[batch_sample, 0, :, :].numpy()), pose_recon[batch_sample, :, :].numpy()))
                    
                # save the last batch input and output of every epoch
                #if im.shape[0] != batch_size:
                if 1:
                #if i == int(len(val_data)/dataloader.batch_size) - 1:
                    num_rows = 8
                    images = im.view(im.shape[0], imageSizeY, imageSizeX).cpu()
                    _,axs = plt.subplots(nrows=1, ncols=8)
                    idx = torch.arange(0,8)

                    # Overlay pose
                    for m in range(0,8):
                        axs[m].imshow(images[idx[m],:,:].cpu(), cmap='gray')
                        axs[m].scatter(pose_recon[idx[m],0,:].cpu(), pose_recon[idx[m],1,:].cpu(), s=0.07, c='green', alpha=0.6)
                        axs[m].grid(False)
                        axs[m].set_axis_off()

                    plt.savefig(output_dir + "/epoch_" + filename[idx[m]][7:] + ".svg")
                    plt.close()

    return reconstructed_pose, corr_coef

train_loss = []
val_loss = []
pose_recon_dict = {}
print("Calculating network predictions",flush=True)
reconstructed_pose, corr_coef = validate(model, val_loader)
print("Finished calculating network predictions")
reconstructed_pose = np.concatenate(reconstructed_pose)
corr_coef = np.array(corr_coef)
corr_coef_dict = {}
corr_coef_dict['corr_coef'] = corr_coef
pose_recon_dict['reconstructed_pose'] = reconstructed_pose
sio.savemat(output_dir + '/predicted_pose.mat', pose_recon_dict)
sio.savemat(output_dir + '/corr_coeff.mat', corr_coef_dict)
print('Results saved, exiting script')
