import torch
import torch.nn as nn
import numpy
from LTC_Model import LTCCell

class LTCFramePredictionModel(nn.Module):

    def __init__(self, sequence_num):
        super(LTCFramePredictionModel, self).__init__()
        self.sequence_num = sequence_num
        # Activation layer
        self.relu = nn.ReLU()
        # Encoder
        self.conv3d0 = nn.Conv3d(in_channels = self.sequence_num, out_channels = self.sequence_num, 
                               kernel_size = 5, stride = 1, padding = 2)
        self.conv3d1 = nn.Conv3d(in_channels = self.sequence_num, out_channels = self.sequence_num, 
                               kernel_size = 5, stride = 1, padding = 2)
        self.conv3d2 = nn.Conv3d(in_channels = self.sequence_num, out_channels = self.sequence_num, 
                               kernel_size = 5, stride = 1, padding = 2)
        self.bn0 = nn.BatchNorm3d(self.sequence_num, track_running_stats = True)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(8, track_running_stats = True)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(16, track_running_stats = True)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(32, track_running_stats = True)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(64, track_running_stats = True)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 4, stride = 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels = 8, out_channels = 3, kernel_size = 4, stride = 2, padding=1)

        # LTC Layers
        self.ltc = LTCCell(num_units=4,_input_shape_one_minus=6272)
        self.x_t = torch.zeros((1,1))


    def forward(self,x):
        batch_size,sequence_len, channels, height, width = x.size()

        # change view to (batch, sequence * channels, height, width) for Convolution
        #x = x.view(batch_size,sequence_len * channels, height, width)
        x = self.conv3d0(x)          # Out (batch, seq * 3, 224, 224)
        x3d0 = x
        x = self.conv3d1(x) + x3d0
        x3d1 = x
        x = self.conv3d2(x) + x3d1
        x = self.bn0(x)
        x = self.relu(x)
        #print("After 3D convolution",x.shape)
        # Change view from (batch, sequence * channels, height, width) to (batch, 3 , height, width)
        x = x.view(x.size(0), 3, -1, x.size(3), x.size(4))
        x = torch.sum(x, dim=2)
        #print("Before Conv1",x.shape)
        # Encode
        x = self.conv1(x)      # Out (batch, 8, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        #print("Before Conv2",x.shape)
        xconv1 = x
        x = self.conv2(x)      # Out (batch, 16, 56, 56)
        x = self.bn2(x)
        x = self.relu(x)
        #print("Before Conv3",x.shape)
        xconv2 = x
        x = self.conv3(x)      # Out (batch, 32, 28, 28)
        x = self.bn3(x)
        x = self.relu(x)
        #print("Before Conv4",x.shape)
        xconv3 = x
        x = self.conv4(x)      # Out (batch, 64, 14, 14)
        x = self.bn4(x)
        x = self.relu(x)
        #print("Before Conv5",x.shape)
        x = self.conv5(x)      # Out (batch, 128, 7, 7)
        x = self.relu(x)

        # Flatten for LTC input
        x = x.view(batch_size,-1)   # In (batch, 128*7*7)
        # LTC Layer
        self.x_t = torch.zeros((6272,1))       
        x = self.ltc(x,self.x_t)    # Out (batch, 6272,16)
        x = x.view(batch_size, 32, 28, 28)
        x = x.relu()
        x = x + xconv3
        # Decode
        x = self.deconv1(x)  #   Out (batch, 16, 56, 56)
        x = self.relu(x)
        x = x + xconv2
        x = self.deconv2(x)  #   Out (batch, 8, 112, 112)
        x = self.relu(x)
        x = x + xconv1
        x = self.deconv3(x)  #   Out (batch, 3, 224, 224)

        return x
    
    def save_model(self, name):
        torch.save(self.state_dict(), name)

model = LTCFramePredictionModel(5)

input = torch.randn(1,5,3,224,224)
output = model(input)