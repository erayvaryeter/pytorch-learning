import copy
import os
import pandas as pd
import csv
import numpy as np
import random
from matplotlib import pyplot as plt
import time

import torch
import torch.optim as optim
from torch import nn
from torch import cuda
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler

class CsvHandler:
    image_directory = ""
    age_csv_file = "age.csv"
    gender_csv_file = "gender.csv"
    race_csv_file = "race.csv"

    def __init__(self, image_directory):
        self.image_directory = image_directory

    def CreateCsvForAge(self):
        with open(self.age_csv_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for filename in os.listdir(self.image_directory):
                file = os.path.join(self.image_directory, filename)
                # checking if it is a file
                if os.path.isfile(file):
                    separation = filename.split("_")
                    data = [filename, int(separation[0])]
                    writer.writerow(data)
        f.close()

    def CreateCsvForGender(self):
        with open(self.gender_csv_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for filename in os.listdir(self.image_directory):
                file = os.path.join(self.image_directory, filename)
                # checking if it is a file
                if os.path.isfile(file):
                    separation = filename.split("_")
                    data = [filename, int(separation[1])]
                    writer.writerow(data)
        f.close()

    def CreateCsvForRace(self):
        with open(self.race_csv_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for filename in os.listdir(self.image_directory):
                file = os.path.join(self.image_directory, filename)
                # checking if it is a file
                if os.path.isfile(file):
                    separation = filename.split("_")
                    data = [filename, separation[2]]
                    writer.writerow(data)
        f.close()

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class DatasetSampler:
    def __init__(self, customDataset, validationSplit = 0.2, shuffle = True):
        self.custom_dataset = customDataset
        self.validation_split = validationSplit
        self.shuffle = shuffle

    def CreateSamplers(self):
        dataset_size = len(self.custom_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        if self.shuffle:
            np.random.seed(random.seed(100))
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        self.train_dataset_size = len(train_indices)
        self.val_dataset_size = len(val_indices)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        samplers = [train_sampler, valid_sampler]
        return samplers

    def GetTrainSize(self): return self.train_dataset_size
    def GetValSize(self): return self.val_dataset_size

def main():
    # create csv files
    image_directory = "D:/Repositories/UTKFace"
    csv_handler = CsvHandler(image_directory)
    csv_handler.CreateCsvForAge()
    csv_handler.CreateCsvForGender()
    csv_handler.CreateCsvForRace()

    # create age dataset
    age_dataset = CustomImageDataset(CsvHandler.age_csv_file, image_directory)
    gender_dataset = CustomImageDataset(CsvHandler.gender_csv_file, image_directory)
    race_dataset = CustomImageDataset(CsvHandler.race_csv_file, image_directory)
    
    # create data samplers
    age_dataset_sampler = DatasetSampler(age_dataset, 0.2, True)
    gender_dataset_sampler = DatasetSampler(gender_dataset, 0.2, True)
    race_dataset_sampler = DatasetSampler(race_dataset, 0.2, True)

    # generate samplers
    age_train_sampler = age_dataset_sampler.CreateSamplers()[0]
    age_validation_sampler = age_dataset_sampler.CreateSamplers()[1]
    gender_train_sampler = gender_dataset_sampler.CreateSamplers()[0]
    gender_validation_sampler = gender_dataset_sampler.CreateSamplers()[1]
    race_train_sampler = race_dataset_sampler.CreateSamplers()[0]
    race_validation_sampler = race_dataset_sampler.CreateSamplers()[1]

    # create data loaders
    batch_size = 8
    age_train_loader = DataLoader(age_dataset, batch_size=batch_size, sampler=age_train_sampler)
    age_validation_loader = DataLoader(age_dataset, batch_size=batch_size, sampler=age_validation_sampler)
    gender_train_loader = DataLoader(gender_dataset, batch_size=batch_size, sampler=gender_train_sampler)
    gender_validation_loader = DataLoader(gender_dataset, batch_size=batch_size, sampler=gender_validation_sampler)
    race_train_loader = DataLoader(race_dataset, batch_size=batch_size, sampler=race_train_sampler)
    race_validation_loader = DataLoader(race_dataset, batch_size=batch_size, sampler=race_validation_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using " + device.__str__() + " device for training")

if __name__ == "__main__":
    main()