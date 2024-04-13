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
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import torch.nn.functional as F
from torch import nn


def train_model_state_of_the_art(model, train_dataloader, val_dataloader ,optimizer, feature_range, device, loss_fn = nn.BCEWithLogitsLoss(), epochs = 10, display_batch_loss_step = 10, display_batch_loss = False , save_folder = "saved_model"):

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

            loss = loss_fn(y_pred['out'], output)

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

                val_loss = loss_fn(y_pred['out'], y)
                loss_val_ep.append(val_loss.cpu().detach().numpy())
            loss_val_all.append(np.mean(loss_val_ep))
            
    ## Print out what's happening
        print(f"Train loss: {loss_train_all[-1]} | Val loss: {[loss_val_all[-1]]}")
        os.makedirs(os.path.join("./saved_models", save_folder), exist_ok=True)
        torch.save(model, os.path.join(os.path.join("./saved_models", save_folder), f'model_{counter}.pth'))
        
    return loss_train_all, loss_val_all


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


# ------------------------------------------ Visualization Methods ------------------------------------------

def visualize_resnet(model1, test_dataloader, device, feature_range, number_gen = 5):
    model1 = model1.to(device)
    model1.eval()
    model1s = []
    outputs = []
    with torch.inference_mode():
        for batch, (data_batch, output_batch) in enumerate(test_dataloader):
            data_batch, output_batch = data_batch.to(device), output_batch.to(device)
            for i in range(len(data_batch)):
                data = data_batch[i].unsqueeze(0)[:, feature_range[0]:feature_range[1],:,:]
                output = output_batch[i].unsqueeze(0)
                if torch.max(output) == 0:
                    continue
                y_pred1 = model1(data)['out']
                model1s.append(y_pred1.cpu().detach().numpy())
                outputs.append(output.cpu().detach().numpy())
                if len(model1s) == number_gen:
                    return model1s, outputs
                
                
    



def createDeepLabv3(inputchannels=4, outputchannels=1, weights=DeepLabV3_ResNet50_Weights.DEFAULT):
    """DeepLabv3 class with custom head
    Args:
        inputchannels (int, optional): The number of input channels in your data. Defaults to 4.
        outputchannels (int, optional): The number of output channels in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet50 backbone.
    """
    model = models.segmentation.deeplabv3_resnet50(weights=weights,
                                                    progress=True)
    
    # Replace the first convolutional layer with a 2D convolutional layer
    model.backbone.conv1 = nn.Conv2d(inputchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    # Adjust the batch normalization layer to match the number of output channels
    model.backbone.bn1 = nn.BatchNorm2d(64)
    
    # Replace the classifier head with a 2D convolutional layer
    # model.classifier = nn.Conv2d(2048, outputchannels, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier = None
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(256, outputchannels, kernel_size=1, stride=1),
        nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
    )

    return model