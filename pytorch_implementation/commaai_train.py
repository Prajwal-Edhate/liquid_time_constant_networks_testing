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
from heatmap import GradCAMSequence
import cv2
import torch.nn.functional as F


class DrivingImageDataset(Dataset):

    def __init__(self, sequence_length, file_path='data/commai', num_data = None, start_point=1100):
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
        self.to_pil = transforms.ToPILImage()
        self.start_point = start_point
        # List for angles and images
        self.steering_angles = []
        self.speeds = []
        self.images = []

        # Open each h5py file and extract data. Discard initial stationary data
        for i in range(num_h5py_files):
            with h5.File(os.path.join(self.file_path,'log',self.h5py_file_names[i]), "r") as file:
                if num_data is not None:
                    steering_angles = file['steering_angle'][self.start_point:self.start_point+num_data]
                    speeds = file['speed'][self.start_point:self.start_point+num_data]
                else:
                    steering_angles = file['steering_angle'][self.start_point*5:][::5]
                    speeds = file['speed'][self.start_point*5:][::5]
                self.steering_angles.append(steering_angles)
                self.speeds.append(speeds)
            
            with h5.File(os.path.join(self.file_path,'camera',self.h5py_file_names[i]), "r") as file:
                if num_data is not None:
                    images = file['X'][self.start_point:self.start_point+num_data]
                else:
                    images = file['X'][self.start_point:]
                self.images.append(images)
        
        self.images = [image for image_list in self.images for image in image_list]
        self.steering_angles = torch.as_tensor([angle for angle_list in self.steering_angles for angle in angle_list])
        self.speeds = torch.as_tensor([speed for speed_list in self.speeds for speed in speed_list])
        # Mean and Std for speed and steering angles
        self.steering_angle_mean = torch.mean(self.steering_angles)
        self.steering_angle_std = torch.std(self.steering_angles)
        self.speed_mean = torch.mean(self.speeds)
        self.speed_std = torch.std(self.speeds)

    def __len__(self):
        return min(len(self.images), len(self.steering_angles)) - self.sequence_length

    def __getitem__(self,index):
        try:
            # get the next steering angle and speed
            if self.steering_angle_std == 0:
                steering_angle = self.steering_angles[index + self.sequence_length]/self.steering_angles[index + self.sequence_length]
            else:
                steering_angle = (self.steering_angles[index + self.sequence_length]- self.steering_angle_mean)/self.steering_angle_std
            
            if self.speed_std == 0:
                speed = self.speeds[index + self.sequence_length]/self.speeds[index + self.sequence_length]
            else:
                speed = (self.speeds[index + self.sequence_length]- self.speed_mean)/self.speed_std

            # Get Image sequence
            images = []
            for i in range(self.sequence_length):
                image = self.images[index + i]
                image = np.transpose(image, (1,2,0))
                image = Image.fromarray(image, mode='RGB')
                image = self.to_tensor(image)
                images.append(image)

            images = torch.stack(images)
            steering_angle = steering_angle.type(torch.float32)
            speed = speed.type(torch.float32)
            speed_and_angle = torch.tensor([speed,steering_angle])

            return images, speed_and_angle
        except IndexError as e:
            print(f"Error in __getitem__ at index {index}: {e}")

    
    def save_sequence(self, start, length, sample_dir ='samples/commaai_sequence'):
        images = self.images[start:start + length]
        images = [self.to_pil(np.transpose(image, (1,2,0))) for image in images]

        os.makedirs(sample_dir, exist_ok=True)

        for i, img in enumerate(images):
            img.save(os.path.join(sample_dir,f'image_{i}.png'))


class SteeringPrediction(nn.Module):

    def __init__(self,sequence_length):
        super(SteeringPrediction, self).__init__()
        self.sequence_length = sequence_length
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5, padding = 2, stride = 2, bias = False)
        self.leakyrelu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(6, affine = True, track_running_stats = True)
        self.conv2 = nn.Conv2d(6, 9, kernel_size = 3, padding = 1, stride = 2, bias = False)
        self.bn2 = nn.BatchNorm2d(9, affine = True, track_running_stats = True)
        self.conv3 = nn.Conv2d(9, 12, kernel_size = 3, padding = 1, stride = 2, bias = False)
        self.bn3 = nn.BatchNorm2d(12, affine = True, track_running_stats = True)
        self.conv4 = nn.Conv2d(12, 15, kernel_size = 3, padding = 1, stride = 2, bias = False)
        self.bn4 = nn.BatchNorm2d(15, affine = True, track_running_stats = True)
        self.conv5 = nn.Conv2d(15, 18, kernel_size = 3, padding = 1, stride = 2, bias = False)
        self.bn5 = nn.BatchNorm2d(18, affine = True, track_running_stats = True)
        self.ltc1 = LTCCell(num_units=16, _input_shape_one_minus = sequence_length * 18)
        self.ltc2 = LTCCell(num_units=2, _input_shape_one_minus = 16)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.state1 = torch.zeros((1,1))
        self.state2 = torch.zeros((1,1))

    def forward(self,x):
        #batch_size, sequence_length, channels, height, width = x.size()
        x = x.permute(1,0,2,3,4)
        intermediate_output = []
        features = []
        for i in range(x.size(0)):
            s = x[i]
            s = self.conv1(s)
            s = self.bn1(s)
            s = self.leakyrelu(s)

            features.append(torch.mean(s, dim = 1, keepdim=True))

            s = self.conv2(s)
            s = self.bn2(s)
            s = self.leakyrelu(s)

            s = self.conv3(s)
            s = self.bn3(s)
            s = self.leakyrelu(s)

            s = self.conv4(s)
            s = self.bn4(s)
            s = self.leakyrelu(s)
            
            s = self.conv5(s)
            s = self.bn5(s)
            s = self.avgpool(s)
            s = s.view(s.size(0),-1)
            # Append Output to list
            intermediate_output.append(s)

        concatenated_out = torch.cat(intermediate_output, dim = 1)
        feat = torch.stack(features)
        feat = feat.permute(1,0,2,3,4)
        #print(feat.shape)
        #print(concatenated_out.shape)
        out = self.ltc1(concatenated_out, self.state1)
        out = self.ltc2(out,self.state2)

        return out, feat
    
    def save_model(self, name):
        torch.save(self.state_dict(), name)


def train(model, dataset, device, num_epochs, batch_size, optimizer, loss_function, train_split, val_split,
          checkpoint_save_dir = 'checkpoint', continue_training = False, sequence_length = 10):

    # Split Dataset in train, val and test dataset
    num_data = len(dataset)
    num_train = int(train_split * num_data)
    num_val = int(val_split * num_data)
    num_test = num_data - num_train - num_val
    print("Inside train checking number of instances", num_data, num_train, num_val, num_test)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    # Load Model weights if continue training
    if continue_training:
        # Get list of weights
        wieght_list = os.listdir(checkpoint_save_dir)
        print("Loading weight - ", wieght_list[-1])
        weights = torch.load(os.path.join(checkpoint_save_dir,wieght_list[-1]))
        temp_in = torch.randn((weights['ltc1.input_w'].size(0),sequence_length, 3, 160, 320))
        temp_out = model(temp_in)
        model.load_state_dict(weights)

    # Send model to GPU
    model.to(device)
    min_val = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequence, targets in train_loader:
            sequence, targets = sequence.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs,_= model(sequence)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequence.size(0)
            del sequence, targets
            torch.cuda.empty_cache()

        running_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequence, targets in val_loader:
                sequence, targets = sequence.to(device), targets.to(device)
                outputs,_= model(sequence)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                # Remove from GPU
                del sequence, targets
                torch.cuda.empty_cache()
            val_loss /= len(val_loader.dataset)
            if val_loss<min_val:
                min_val = val_loss
                model.save_model(os.path.join(checkpoint_save_dir,f"checkpoint_min_val_loss.pkl"))
        
        print(f"Epoch {epoch}/{num_epochs}: Train_loss {running_loss} Val_loss {val_loss}")

        # Test (Save examples in folder along with sequence, prediction and target)

        if (epoch % 5 == 0):
            model.save_model(os.path.join(checkpoint_save_dir,f"checkpoint_{epoch}.pkl"))
            test_loss = 0.0
            with torch.no_grad():
                for sequence, targets in test_loader:
                    sequence, targets = sequence.to(device), targets.to(device)
                    outputs,_ = model(sequence)
                    loss = loss_function(outputs, targets)
                    test_loss += loss.item()
                    # Remove from GPU
                    del sequence, targets
                    torch.cuda.empty_cache()
                test_loss /= len(val_loader.dataset)
            
            print(f"Eval Epoch {epoch}/{num_epochs}: Test_loss {test_loss}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    
    sequence_length = 24
    start_point = 1000
    checkpoint_path = 'checkpoint/commaai_steering'
    os.makedirs(checkpoint_path, exist_ok=True)
    train_split = 0.8
    val_split = 0.1

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset
    dataset = DrivingImageDataset(sequence_length=sequence_length, start_point=start_point, num_data=None)

    # Get Model
    model = SteeringPrediction(sequence_length=sequence_length)
    num_params = count_parameters(model)
    print("Number of parameters:", num_params)

    # Define training parameters
    num_epochs = 100
    batch_size = 64
    lr = 0.001

    # Define Loss function and Optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = lr)

    train(model=model, dataset=dataset, device=device, num_epochs=num_epochs,
        batch_size=batch_size, optimizer=optimizer, loss_function=loss_function, train_split=train_split, val_split=val_split,
        checkpoint_save_dir=checkpoint_path, continue_training= False)

def heatmap():
    # Load Model Weights
    model = SteeringPrediction(24)
    weights = torch.load('checkpoint/commaai_steering/checkpoint_1_min_val_loss.pkl')
    temp_input = torch.randn((weights['ltc1.input_w'].size(0),24,3,160,320))
    _,_ = model(temp_input)
    model.load_state_dict(weights)
    model.eval()
    # Get Driving dataset for heatmap generation
    dataset = DrivingImageDataset(sequence_length=24, num_data=200)
    input = dataset[180][0].unsqueeze(0)
    _,heatmap = model(input)
    #print(heatmap.shape)
    with torch.no_grad():
        for i in range(heatmap.size(0)):
            for j in range(heatmap.size(1)):
                heat_full = F.interpolate(heatmap[i][j].unsqueeze(0), size = (160,320),
                                          mode = 'bilinear', align_corners = False)
                heatmap_np = heat_full.squeeze().numpy()
                # Normalize heat map
                heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
                heatmap_np = (heatmap_np * 255).astype(np.uint8)

                # Create PIL Image
                heatmap_color = cv2.applyColorMap((heatmap_np), cv2.COLORMAP_JET)
                original = transforms.ToPILImage()(input[i][j])
                #print(type(original))
                # Save the images
                sequence_directory = f'heatmap/sequence_{i+1}' if heatmap.size(0) > 0 else 'heatmap/sequence'
                os.makedirs(sequence_directory, exist_ok=True)
                cv2.imwrite(f'{sequence_directory}/map_{j}.jpg',heatmap_color)
                original.save(f'{sequence_directory}/original_{j}.jpg')

if __name__ == '__main__':
    main()

