import torch
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
from LTC_model import LTCCell
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py as h5
from PIL import Image
import torchvision.transforms as transforms

class DrivingImageDataset(Dataset):

    def __init__(self, sequence_length, file_path, num_data = None):
        super(DrivingImageDataset, self).__init__()
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.num_data = num_data

        # Get number of h5py files
        self.h5py_file_names = os.listdir(os.path.join(self.file_path,'log'))
        num_h5py_files = len(self.h5py_file_names)

        # set standard deviation and mean for images
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(image_mean,image_std)])
        
        # List for angles and images
        self.steering_angles = []
        self.images = []

        # Open each h5py file and extract data. Discard initial stationary data
        for i in range(num_h5py_files):
            with h5.File(os.path.join(self.file_path,'log',self.h5py_file_names[i]), "r") as file:
                if num_data is not None:
                    steering_angles = file['steering_angle'][3000:3000+num_data]
                else:
                    steering_angles = file['steering_angle'][3000:]
                self.steering_angles.append(steering_angles)
            
            with h5.File(os.path.join(self.file_path,'camera',self.h5py_file_names[i]), "r") as file:
                if num_data is not None:
                    images = file['X'][1100:1100+num_data]
                else:
                    images = images[1100:]
                self.images.append(images)
        
        self.images = [image for image_list in self.images for image in image_list]
        self.steering_angles = torch.as_tensor([angle for angle_list in self.steering_angles for angle in angle_list])
        self.steering_angle_mean = torch.mean(self.steering_angles)
        self.steering_angle_std = torch.std(self.steering_angles)


    def __getitem__(self,index):
        # get the next steering angle
        if self.steering_angle_std == 0:
            steering_angle = self.steering_angles[index + self.sequence_length]/self.steering_angles[index + self.sequence_length]
        else:
            steering_angle = (self.steering_angles[index + self.sequence_length]- self.steering_angle_mean)/self.steering_angle_std
        images = self.images[index: index + self.sequence_length]
        images = [self.to_tensor(Image.fromarray(np.transpose(image,(1,2,0)), mode = 'RGB')) for image in images]
        return images, steering_angle

    def __len__(self):
        return len(self.steering_angles) - self.sequence_length



