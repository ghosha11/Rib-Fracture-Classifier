#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import necessary libraries
import argparse
import pandas as pd
import nibabel as nib
import os 
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import cv2
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch
import torchmetrics
import copy
import shutil


# In[2]:

# Set up command-line argument parser
my_parser = argparse.ArgumentParser(description='size of the image to resize')
my_parser.add_argument('size',
                       metavar='size',
                       type=int,
                       help='size of the image to resize')
args = my_parser.parse_args()
size = args.size
print(size)


# In[3]:

# Set image size, batch size, and device (GPU)
#size=448
batchsize=8
device="cuda"

# Create directories to save model checkpoints and best models
try:
    os.mkdir("./checkpoint_dn121_CXNTL_H/")
except: pass
try:
    os.mkdir("./best_model_dn121_CXNTL_H/")
except: pass


# In[4]:

# Define the path to the CSV file containing data information
path_csv="./train_test_split.csv"


# In[5]:

# Load the CSV data into a pandas dataframe
data=pd.read_csv(path_csv)


# In[6]:

# Define the paths for the two classes (NoRibFracture and RibFracture)
nrf="./NoRibFracture/"
rf="./RibFracture/"


# In[7]:

# Update the image paths in the dataframe to include the appropriate class directories
data.loc[data.label==0,"image"]=data[data.label==0].image.apply(lambda x : os.path.join(nrf,x))
data.loc[data.label==1,"image"]=data[data.label==1].image.apply(lambda x :os.path.join(rf,x))


# In[8]:


# Define a custom dataset class for the rib dataset
class RibDataset(Dataset):
    
    def __init__(self,dataframe, transform=None,data_type="train"):
   # Set the paths to the images and corresponding labels 
        self.full_filenames = list(dataframe.image)
        self.labels = list(dataframe.label)
        self.transform = transform


    def __len__(self):
        # return size of dataset 
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label 
        nii=nib.load(self.full_filenames[idx])
        image= nii.get_fdata()[:,:,0]
        #image=(image-0.49)/0.248
        image = cv2.resize(image,(size, size)).astype(np.float32)
        if self.transform!=None:
            image = self.transform(image)
        return image, self.labels[idx]


# In[9]:

# Define the image transforms for training and validation

train_transforms = transforms.Compose([
                                      # Use mean and std from preprocessing notebook
                                    transforms.ToPILImage(),
                                    transforms.RandomAffine( # Data Augmentation
                                                degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                    transforms.RandomResizedCrop((size, size), scale=(0.8, 1)),
                                    transforms.ToTensor(), 
                                    torchvision.transforms.Lambda(lambda x: x.repeat(3,1,1)),# Convert numpy array to tensor
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

    

])

val_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                   torchvision.transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])


# In[10]:


train=data[data.split=="train"]
test=data[data.split=="test"]
valid=data[data.split=="val"]


# In[11]:

# Creating RibDataset instances for training and validation data

train_data=RibDataset(dataframe=train,transform=train_transforms)
valid_data=RibDataset(dataframe=valid,transform=val_transforms)


# In[12]:
# Creating data loaders for training and validation data
train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True) 
val_dl = DataLoader(valid_data, batch_size=batchsize, shuffle=False)


# In[13]:

# Storing the data loaders in a dictionary

dataloaders_dict = {'train': train_dl,
                   'val':val_dl}


# In[14]:

# Printing the number of images in the training and validation datasets

print(f"There are {len(train_data)} train images and {len(valid_data)} val images")


# In[15]:


class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = nn.functional.softmax(self.densenet121(x),dim=-1)
        return x


# In[16]:


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


# In[17]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,start_epochs=0,
                valid_loss_min_input=0,checkpoint_path="./", best_model_path="./",
                df=pd.DataFrame(columns=['val_acc_history','val_auc_history', 'train_auc_history','train_acc_history','train_loss', 'valid_loss'])
               ):
    val_acc_history = []
    val_auc_history=[]
    train_auc_history=[]
    train_acc_history=[]
    val_accuracy = torchmetrics.Accuracy().to(device)
    train_accuracy = torchmetrics.Accuracy().to(device)
    train_AUC=torchmetrics.AUROC(num_classes=2).to(device)
    val_AUC=torchmetrics.AUROC(num_classes=2).to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epochs,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        output_train=[]
        output_val=[]
        target_train=[]
        target_val=[]

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                   

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    if phase == 'train':
                        output_train.append(outputs)
                        target_train.append(labels)
                        
                    else:
                        output_val.append(outputs)
                        target_val.append(labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase=='train':
                batch_train_acc=train_accuracy(torch.cat(output_train, dim=0),torch.cat(target_train, dim=0).type(torch.int))
                batch_train_auc=train_AUC(torch.cat(output_train, dim=0),torch.cat(target_train, dim=0).type(torch.int))
                epoch_train_acc=train_accuracy.compute()
                epoch_train_auc=train_AUC.compute()
                print("train accuracy is{:.4f} AUC is {:.4f}". format(float(epoch_train_acc),float(epoch_train_auc)))
                train_accuracy.reset()
                train_AUC.reset()
                df.loc[epoch,"train_loss"]=epoch_loss
            elif phase=='val':
                batch_val_acc=val_accuracy(torch.cat(output_val, dim=0),torch.cat(target_val, dim=0).type(torch.int))
                batch_val_auc=val_AUC(torch.cat(output_val, dim=0),torch.cat(target_val, dim=0).type(torch.int))
                epoch_val_acc=val_accuracy.compute()
                epoch_val_auc=val_AUC.compute()
                print("val accuracy is {:.4f} AUC is {:.4f}".format(float(epoch_val_acc),float(epoch_val_auc)))
                val_accuracy.reset()
                val_AUC.reset()
                df.loc[epoch,"val_loss"]=epoch_loss
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_val_auc> best_acc:
                best_acc = epoch_val_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint = {
                    'epoch': epoch + 1,
                    'valid_loss_min': epoch_val_auc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'df':df
                    }
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        
        val_acc_history.append(float(epoch_val_acc))
        val_auc_history.append(float(epoch_val_auc))
        train_acc_history.append(float(epoch_train_acc))
        train_auc_history.append(float(epoch_train_auc))
        df.loc[epoch,"val_acc_history"]=float(epoch_val_acc)
        df.loc[epoch,"val_auc_history"]=float(epoch_val_auc)
        df.loc[epoch,"train_acc_history"]=float(epoch_train_acc)
        df.loc[epoch,"train_auc_history"]=float(epoch_train_auc)
        scheduler.step(epoch_val_auc)
        
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': epoch_val_auc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'df':df
        }
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        print()
        

    print('Best val AUC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc ,df


# In[20]:

# Defining a list of learning rates and weight decays for hyperparameter tuning

lr=[0.001, 0.0001, 0.00001,0.1, 0.01]
wd=[0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
#lr=[ 0.001]
#wd=[0.0]

# Creating an empty DataFrame to store hyperparameter tuning results

hypertuning=pd.DataFrame(columns=["size","lr","wd","best_auc"])
# Initializing a counter for combination number

c=0

# Looping over learning rates and weight decays

for learning_rate in lr:
    for weight_decay in wd:
        print("first combination number{}".format(c))
        # Creating the checkpoint and best model paths based on the current combination

        p="_".join(["size",str(size),"learning_rate",str(learning_rate),"weight_decay",str(weight_decay)])
        checkpoint_path="./checkpoint_dn121_CXNTL_H/current_dn121_CXNTL_1_"+p+".pth"
        best_model_path="./best_model_dn121_CXNTL_H/best_dn121_CXNTL_1_"+p+".pth"
        # Creating an instance of the DenseNet121 model and loading pre-trained weights

        model_1c=DenseNet121(14,True)
        checkpoint = torch.load("./m-25012018-123527.pth.tar")
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            new_key = key[7:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            if (new_key.split(".")[-2].isdigit()) and len(new_key.split("."))>4:
                    l=new_key.split(".")
                    l[-3]=l[-3]+l[-2]
                    l.remove(l[-2])
                    new_key1=".".join(l)
                    state_dict[new_key1] = state_dict[new_key]
                    del state_dict[new_key]
        model_1c.load_state_dict(state_dict, strict=False)
        model_1c.densenet121.classifier[0]=nn.Linear(in_features=1024, out_features=2, bias=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_1c = model_1c.to(device)
        optimizer_ft = optim.SGD(model_1c.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max',
        factor=0.1, patience=5, threshold=0.0001, threshold_mode='abs',verbose=True)
        criterion = nn.CrossEntropyLoss()
        # Training the model, saving checkpoints, and obtaining the best accuracy

        model_ft,best_acc,df = train_model(model_1c, dataloaders_dict, criterion, optimizer_ft, num_epochs=50,
                                      checkpoint_path=checkpoint_path,
                                      best_model_path=best_model_path
                                      )
        # Storing the hyperparameters and best accuracy in the DataFrame
     
        hypertuning.loc[c,'size']=size
        hypertuning.loc[c,'lr']=learning_rate
        hypertuning.loc[c,'wd']=weight_decay
        hypertuning.loc[c,'best_auc']=float(best_acc.cpu())
        hypertuning.to_csv("./densenet_448_hypertuning.csv")
        c+=1


# In[ ]:





# In[ ]:




