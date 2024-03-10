import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from LTC_model import LTCCell
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.autograd.set_detect_anomaly(True)

class OccupancyData(Dataset):

    def __init__(self, sequence_length, file_path, inc=1,mean = torch.zeros(5), std = torch.ones(5)):
        super(OccupancyData, self).__init__()
        self.file_path =  file_path
        self.sequence_length = sequence_length
        self.inc = inc

        self.df = pd.read_csv(self.file_path)

        self.data_x = np.stack([
            self.df['Temperature'].values,
            self.df['Humidity'].values,
            self.df['Light'].values,
            self.df['CO2'].values,
            self.df['HumidityRatio'].values,
        ], axis = -1)

        self.data_y = self.df['Occupancy'].values.astype(np.float64)

        if 'datatraining' in file_path:
            self.mean_values = torch.as_tensor(np.mean(self.data_x, axis=0))
            self.std_values = torch.as_tensor(np.std(self.data_x, axis=0))
        else:
            self.mean_values = mean
            self.std_values = std

    def __getitem__(self,index):
        seq = (torch.as_tensor(self.data_x[index : index + self.sequence_length])-torch.as_tensor(self.mean_values))/torch.as_tensor(self.std_values)
        target = torch.as_tensor(self.data_y[index])

        return seq, target
    
    def __len__(self):
        return len(self.data_y) - self.sequence_length
    
class PredictionNetwork(nn.Module):

    def __init__(self, sequence_length):
        super(PredictionNetwork, self).__init__()
        self.sequence_length = sequence_length
        self.ltc = LTCCell(num_units=32, _input_shape_one_minus=sequence_length * 5)
        self.ltc2 = LTCCell(num_units=1, _input_shape_one_minus=32)
        self.state = torch.zeros((1,1))
        self.state2 = torch.zeros((1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size,self.sequence_length * 5)
        x = self.ltc(x, self.state)
        x = x.view(batch_size,32)
        x = self.ltc2(x, self.state2)
        x = self.sigmoid(x)
        x = x.view(batch_size)
        return x
    
    def save_model(self, name):
        torch.save(self.state_dict(), name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, device, num_epochs, batch_size, optimizer, loss_function, train_dataset, val_dataset, test_dataset,
          checkpoint_save_dir = 'checkpoint'):
    
    # Create checkpoint and sample directory if it does not exist
    if not os.path.exists(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)
    


    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    # Send model to GPU
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequence, targets in train_loader:
            sequence, targets = sequence.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs= model(sequence)
            #print(outputs, targets)
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
                outputs= model(sequence)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                # Remove from GPU
                del sequence, targets
                torch.cuda.empty_cache()
            val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch}/{num_epochs}: Train_loss {running_loss} Val_loss {val_loss}")

        # Test (Save examples in folder along with sequence, prediction and target)

        if (epoch % 5 == 0):
            model.save_model(os.path.join(checkpoint_save_dir,f"checkpoint_{epoch}.pkl"))
            test_loss = 0.0
            with torch.no_grad():
                for sequence, targets in test_loader:
                    sequence, targets = sequence.to(device), targets.to(device)
                    outputs = model(sequence)
                    loss = loss_function(outputs, targets)
                    test_loss += loss.item()
                    # Remove from GPU
                    del sequence, targets
                    torch.cuda.empty_cache()
                test_loss /= len(val_loader.dataset)
            
            print(f"Eval Epoch {epoch}/{num_epochs}: Test_loss {test_loss}")

def main():

    filepath = 'data/occupancy'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    sequence_length = 10
    num_epochs = 100
    lr = 0.005

    # Get data
    train_dataset = OccupancyData(sequence_length=sequence_length,file_path=os.path.join(filepath,'datatraining.txt'))
    mean = train_dataset.mean_values
    std = train_dataset.std_values
    test0_dataset = OccupancyData(sequence_length=sequence_length,file_path=os.path.join(filepath,'datatest.txt'),mean = mean, std = std)
    test1_dataset = OccupancyData(sequence_length=sequence_length,file_path=os.path.join(filepath,'datatest2.txt'), mean = mean, std = std)

    # Get model
    model = PredictionNetwork(sequence_length=sequence_length)
    num_params = count_parameters(model)
    print("Number of parameters:", num_params)

    # Define Loss function and Optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = lr)

    train(model=model, device=device, num_epochs=num_epochs, batch_size=batch_size, optimizer=optimizer,
          loss_function=loss_function,train_dataset=train_dataset,val_dataset=test0_dataset,test_dataset=test1_dataset)


if __name__ == '__main__':
    main()
