import torch
import torch.nn as nn
from FrameModel import LTCFramePredictionModel
from FramePrediction2 import Frame
from torchvision import transforms
from PIL import Image
import natsort
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.autograd.set_detect_anomaly(True)

class FrameDataset(Dataset):

    def __init__(self, frame_path = 'export', window_size = 6, num_data = 510):
        super(FrameDataset, self).__init__()
        self.frame_path = frame_path
        self.window_size = window_size
        self.num_data = num_data
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor()])
      
        # Get the names of frame (.jpg) from export folder using os
        self.frame_names = sorted([file for file in os.listdir(self.frame_path) if file.endswith(".jpg")])

        # Get every second file name to eleiminate duplication
        self.frame_names = self.frame_names[::2][:self.num_data]
    
    def __len__(self):
        return len(self.frame_names)-self.window_size
        
    def __getitem__(self, index):
        image_sequence = []
        
        # Get the image sequence starting index i to i + sequence_len
        sequence_names = self.frame_names[index:index+self.window_size]
        prediction_frame_name = self.frame_names[index + self.window_size]

        # Get Image sequence and target frame
        for i in range(len(sequence_names)):
            image = Image.open(os.path.join(self.frame_path, sequence_names[i])).convert('RGB')
            image = self.transform(image)

            image_sequence.append(image)
        
        prediction_frame = Image.open(os.path.join(self.frame_path, prediction_frame_name)).convert('RGB')
        prediction_frame = self.transform(prediction_frame)
        image_sequence = torch.stack(image_sequence)

        return image_sequence, prediction_frame

def check_sum(train_split, val_split):
    if train_split + val_split >1:
        raise ValueError("The sum of train_split and val_split cannot be greater than 1.")
    
def save_image(sequence, targets, outputs, epoch,sample_dir = 'samples'):
    transform_to_Image = transforms.ToPILImage()
    
    # Extract and save sequence, ground_truth and predictions in seperate folders with epoch name
    for i in range(targets.size(0)):
        # Create dir for each sequence sample
        seq_path = os.path.join(sample_dir,str(epoch), f"seq_{i}")
        if not os.path.exists(seq_path):
            os.makedirs(os.path.join(sample_dir,str(epoch), f"seq_{i}"))
        # Extract sequence images and convert to PIL Image
        input_images = [transform_to_Image(sequence[i][j]) for j in range(sequence[i].size(0))]
        for k, image in enumerate(input_images):
            image.save(os.path.join(seq_path, f"input_{k}.jpg"))
        ground_truth = transform_to_Image(targets[i])
        ground_truth.save(os.path.join(seq_path,"ground_truth.jpg"))
        prediction = transform_to_Image(outputs[i])
        prediction.save(os.path.join(seq_path,f"prediction.jpg"))
        


def train(model, dataset, device, num_epochs, batch_size, optimizer, loss_function, train_split, val_split,
          checkpoint_save_dir = 'checkpoint', sample_dir = 'samples'):

    # Check if train + validation set is less than total dataset
    try:
        check_sum(train_split=train_split, val_split=val_split)
    except ValueError as e:
        print(e)
    
    # Create checkpoint and sample directory if it does not exist
    if not os.path.exists(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Split Dataset in train, val and test dataset
    num_data = len(dataset)
    num_train = int(train_split * num_data)
    num_val = int(val_split * num_data)
    num_test = num_data - num_train - num_val

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    # Send model to GPU
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequence, targets in train_loader:
            sequence, targets = sequence.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs= model(sequence)
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
                    save_image(sequence=sequence[:2], targets=targets[:2], 
                               sample_dir=sample_dir, epoch=epoch, outputs=outputs[:2])
                    # Remove from GPU
                    del sequence, targets
                    torch.cuda.empty_cache()
                test_loss /= len(val_loader.dataset)
            
            print(f"Eval Epoch {epoch}/{num_epochs}: Test_loss {test_loss}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    torch.cuda.empty_cache()
    frame_path= 'export'

    # Define GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset
    window_size = 3
    num_data = 1030
    frame_dataset = FrameDataset(window_size=window_size, num_data=num_data)

    # Define Model
    model = LTCFramePredictionModel(sequence_num=window_size)
    num_params = count_parameters(model)
    print("Number of parameters:", num_params)

    # Define training parameters
    num_epochs = 100
    batch_size = 64
    lr = 0.001

    # Define Loss function and Optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = lr)
    
    train(model=model, dataset=frame_dataset, device=device, num_epochs=num_epochs,
          batch_size=batch_size, optimizer=optimizer, loss_function=loss_function, train_split=0.8, val_split=0.1)



if __name__ == '__main__':
    main()