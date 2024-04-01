import os
import glob
import random
import numpy as np
import nibabel as nib
from torch.utils.data import random_split
import matplotlib.pyplot as plt    
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

import torch.nn.functional as F
from torch import nn

def generate_datapoint(src,files, size):
    if len(files) != 5:
        return None, False
    datapoint = []
    aug1 = []
    aug2 = []
    aug3 = []
    aug4 = []
    randangle1 = random.randint(0, 360)
    randangle2 = random.randint(0, 360)
    randangle3 = random.randint(0, 360)
    randangle4 = random.randint(0, 360)
    for file in sorted(files):
        new_file = create_smaller_image(src, file)
        datapoint.append(new_file)
        aug1.append(rotate_image(new_file, randangle1))
        aug2.append(rotate_image(new_file, randangle2))
        aug3.append(rotate_image(new_file, randangle3))
        aug4.append(rotate_image(new_file, randangle4))
    numpy_datapoint = np.stack(datapoint)
    return (numpy_datapoint, aug1, aug2, aug3, aug4), True

def generate_dataset(dataset_path, destination_path, size):
    counter = 0
    os.makedirs(destination_path, exist_ok=True)
    for src, dirs, files in os.walk(dataset_path):
        datapt, is_generated  = generate_datapoint(src, files, size)
        if is_generated:
            for i in range(len(datapt)):
                np.save(os.path.join(destination_path, f"{i}_{src.split('/')[-1]}"), datapt[i])
                counter += 1
    print(f"Generated {counter} datapoints")
    
def rotate_image(img, angle):
    image = torch.from_numpy(img)
    image = image.unsqueeze(0).float()
    rotation_transform = transforms.Compose([
        transforms.Lambda(lambda x: TF.rotate(x, angle))
    ])

    rotated_tensor = torch.stack([rotation_transform(image[:, i]) for i in range(image.shape[1])])
    
    rotated_tensor = rotated_tensor.squeeze(1)
    rotated_np_array = rotated_tensor.numpy()

    return rotated_np_array
    
def create_smaller_image(path,file_name,jump=10):
    smaller_image = np.zeros((155// jump + 1, 240, 240), np.float32)
    image = nib.load(os.path.join(path, file_name)).get_fdata()
    i = 0
    for s in range(0, image.shape[2], jump):
        if image[:,:,s].max() != 0:
            smaller_image[i] = image[:, :, s] / image[:, :, s].max()
        else:
            smaller_image[i] = image[:, :, s]
        i += 1
    return smaller_image

def load_dataset(dataset_path):
    outputs = []
    for src, dirs, files in os.walk(dataset_path):
        print(src)
        for file in files:
            file_path = os.path.join(src, file)
            output = np.load(file_path)
            outputs.append(output)
    return outputs

def split_input_output(dataset):
    data = []
    for datapoint in dataset:
        first_array = np.array([datapoint[0], *datapoint[-3:]])
        second_array = np.array(datapoint[1:2])
        data.append((first_array, second_array))
    return data

def split_layers(data):
    shape = data.shape
    newdata = data.transpose(0, 2, 1, 3, 4)
    output = newdata.reshape(shape[0]*16, 5, 240, 240)
    return output


def display_slice(slice1, slice2, tag1 = 'Slice 1', tag2 = 'Slice 2'):
    # if slice_index < 0 or slice_index >= array_3d.shape[2]:
    #     raise ValueError("Slice index is out of bounds.")
    
    # Extract the specified slice
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    
    # Display the first slice
    ax[0].imshow(slice1)
    ax[0].set_title(tag1)
    ax[0].axis('off')  
    
    # Display the second slice
    ax[1].imshow(slice2)
    ax[1].set_title(tag2)
    ax[1].axis('off') 
    
    plt.show()
    
    
    

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (np.array([sample[0], *sample[-3:]]), np.array([sample[1]]))
    
    
    
# ------------------------------------------ Training Methods ------------------------------------------
def train_model(model, train_dataloader, val_dataloader ,optimizer, feature_range, device, loss_fn = nn.BCEWithLogitsLoss(), epochs = 10, display_batch_loss_step = 10, display_batch_loss = False , save_folder = "saved_model"):

    model = model.to(device)
    
    loss_train_all = []
    loss_val_all = []
    counter = 0
    
    for epoch in range(epochs):
        counter += 1
        print(f"Epoch: {epoch+1} of {epochs}")
        model.train()
        loss_ep = []
        
        for batch, (data, output) in enumerate(train_dataloader):
            data, output = data.to(device), output.to(device)
            data = data[:, feature_range[0]:feature_range[1],:,:]

            y_pred = model(data)

            loss = loss_fn(y_pred, output)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            loss_ep.append(loss.cpu().detach().numpy())
            
            if display_batch_loss and batch % display_batch_loss_step == 0:
                print(f"Batch: {batch+1} of {len(train_dataloader)} Loss: {loss}")
                
        loss_train_all.append(np.mean(loss_ep))
        
        model.eval()
        with torch.inference_mode():
            loss_val_ep = []
            for val, y in val_dataloader:
                val, y = val.to(device), y.to(device)
                val = val[:, feature_range[0]:feature_range[1],:,:]

                y_pred = model(val)

                val_loss = loss_fn(y_pred, y)
                loss_val_ep.append(val_loss.cpu().detach().numpy())
            loss_val_all.append(np.mean(loss_val_ep))
            
    ## Print out what's happening
        print(f"Train loss: {loss_train_all[-1]} | Test loss: {[loss_val_all[-1]]}")
        os.makedirs(os.path.join("./saved_models", save_folder), exist_ok=True)
        torch.save(model, os.path.join(os.path.join("./saved_models", save_folder), f'model_{counter}.pth'))
        
    return loss_train_all, loss_val_all

def dice_loss(pred, target, smooth=1.):
    pred = F.sigmoid(pred)

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.*intersection + smooth)/(pred.sum() + target.sum() + smooth)

    return 1 - dice


def plot_two_lines_same_x(y1, y2):
    if len(y1) != len(y2):
        raise ValueError("The lists must have the same length.")

    x = list(range(len(y1)))  # Generates an x-axis based on the length of the y-values lists

    plt.plot(x, y1, label='train')
    plt.plot(x, y2, label='val')
    plt.legend()
    plt.show()
    
    
    
def test_model(model, test_dataloader, device, lossfn, store_preds=20):
    model = model.to(device)
    model.eval()
    losses = []
    preds = []
    actuals = []
    with torch.inference_mode():
        for batch, (data, output) in enumerate(test_dataloader):
            if batch % 10 == 0: print(f"{batch}/{len(test_dataloader)}")
            data, output = data.to(device), output.to(device)
            y_pred = model(data)
            if len(preds) < store_preds:
                for i in range(len(output)):
                    if torch.max(output[i]) == 0:
                        continue
                    preds.append(y_pred[i].cpu().detach().numpy())
                    actuals.append(output[i].cpu().detach().numpy())
            losses.append(lossfn(y_pred, output).cpu().detach().numpy())
    return np.mean(losses), np.array(preds), np.array(actuals)


def visualize_model_single_channels(model1,model2,model3, test_dataloader, device, feature_range, number_gen = 5):
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model1.eval()
    model2.eval()
    model3.eval()
    model1s = []
    model2s = []
    model3s = []
    outputs = []
    with torch.inference_mode():
        for batch, (data_batch, output_batch) in enumerate(test_dataloader):
            data_batch, output_batch = data_batch.to(device), output_batch.to(device)
            for i in range(len(data_batch)):
                data = data_batch[i].unsqueeze(0)[:, feature_range[0]:feature_range[1],:,:]
                output = output_batch[i].unsqueeze(0)
                if torch.max(output) == 0:
                    continue
                y_pred1 = model1(data)
                y_pred2 = model2(data)
                y_pred3 = model3(data)
                model1s.append(y_pred1.cpu().detach().numpy())
                model2s.append(y_pred2.cpu().detach().numpy())
                model3s.append(y_pred3.cpu().detach().numpy())
                outputs.append(output.cpu().detach().numpy())
                if len(model1s) == number_gen:
                    return model1s, model2s, model3s, outputs
                
                
                

def display_4_slice(slice1, slice2, slice3, slice4, tag1 = 'Slice 1', tag2 = 'Slice 2', tag3 = 'Slice 3', tag4 = 'Slice 4'):
    fig, ax = plt.subplots(1, 4, figsize=(10, 5)) 
    
    ax[0].imshow(slice1)
    ax[0].set_title(tag1)
    ax[0].axis('off')  
    
    ax[1].imshow(slice2)
    ax[1].set_title(tag2)
    ax[1].axis('off') 
    
    ax[2].imshow(slice3)
    ax[2].set_title(tag3)
    ax[2].axis('off') 
    
    ax[3].imshow(slice4)
    ax[3].set_title(tag4)
    ax[3].axis('off') 
    
    plt.show()
    