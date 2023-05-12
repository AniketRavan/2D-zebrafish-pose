import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, img_files_address, transform=None):
        self.img_files_address = img_files_address
        self.data_size = len(img_files_address)
        self.transform = transform
    
    def load_image(self,path):
        return torch.tensor(cv2.imread(path))
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = Image.open(self.img_files_address[idx])
        image = self.transform(image)
        filename = self.img_files_address[idx][-14:-4]
        return image, filename
