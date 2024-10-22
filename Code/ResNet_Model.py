# Graphing / Math Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Statistical Analysis Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms, models

# Miscellaneous
from datetime import datetime
import platform
import miscellaneous as misc

def loadAndSortImages(filePath, trainPercentage, testPercentage): #Returns a list of datasets
  
  adjustedPath = (misc.adjustFilepathsForOS(filePath))[0]
  
  #Set parameters for loading images
  transform = transforms.Compose([
      
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
  ])
  
  #Load images
  dataset = datasets.ImageFolder(root=adjustedPath, transform=transform)
  
  trainingDatasetSize = int(np.ceil(trainPercentage*len(dataset)))
  testingDatasetSize = int(np.floor(testPercentage*len(dataset)))

  trainingDataset, testingDataset = random_split(dataset, [trainingDatasetSize, testingDatasetSize])
  
  trainingDataloader = DataLoader(trainingDataset, batch_size = 32, shuffle = True), 
  testingDataloader = DataLoader(testingDataset, batch_size = 32, shuffle = False)
  

  return [trainingDataloader, testingDataloader]

def trainModel(imageDataset):

  resNet34 = torchvision.models.resnet(pretrained=True)


def main():
    
  filePath = ['../Datasets/Dataset 1/Dataset 1/Colorectal Cancer']
  imageDataLoader = loadAndSortImages(filePath,0.7, 0.3)

if __name__ == "__main__":
    main()

