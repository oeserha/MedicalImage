# Dataloader & Box Integration

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os

#TODO: edit path
PATH = "/home/haleigh/Documents/Segmentations/"

# Access MRI data folder
def get_mri_data(path = None):
    if path is not None:
        FILE_PATH = path
    else:
        FILE_PATH = 'seg_list.xlsx'

    mri_data = pd.read_excel(FILE_PATH, sheet_name="Completed Segmentations")
    return mri_data

# Additional Fields for MRI Data
def clean_mri_data(mri_data):
    mri_data['Total Images'] = mri_data['Number of Slices'] + mri_data['Number of Brightness Levels']
    mri_data['Start_Index'] = 0
    mri_data['Has MRI'] = ~mri_data['PNG filtered MRI'].isna()
    mri_data['Has Seg'] = ~mri_data['PNG segmentation'].isna()
    mri_data = mri_data[(mri_data['Has MRI']) & (mri_data['Has Seg'])]

    mri_data.rename(columns={"MRI/Patient ID": "patient_id"}, inplace=True)
    mri_data = mri_data.reset_index()
    mri_dummy = "MRI PNGs"

    folders = []
    for i in range(len(mri_data)):
        brightness_folders = os.listdir(rf"{PATH}{mri_data['patient_id'][i]}/{mri_dummy}")
        folders.append(brightness_folders)

    mri_data['Brightness Folders'] = folders

    return mri_data

def train_test(mri_data):
    # Train/Test split
    train_data, test_data = train_test_split(mri_data, test_size=0.25)
    train_data = train_data.reset_index().drop(columns = 'index')
    test_data = test_data.reset_index().drop(columns = 'index')

    for i in range(len(train_data)):
        train_data.loc[i, 'Start_Index'] = sum(train_data['Total Images'][:i])
        
    for i in range(len(test_data)):
        test_data.loc[i, 'Start_Index'] = sum(test_data['Total Images'][:i])

    return train_data, test_data

class CancerDataset(Dataset):
    def __init__(self, labels, path = PATH, train = True, transform=None, target_transform=None):
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.path = path

    def __len__(self):
        len = sum(self.img_labels['Total Images'])
        return int(len)

    def __getitem__(self, idx):
        row = self.img_labels.loc[self.img_labels.Start_Index.where(self.img_labels.Start_Index >= idx).first_valid_index()]
        
        patient = row['patient_id']
        bright_level = int((idx - row['Start_Index']) % len(row['Brightness Folders']))
        bright_id = row['Brightness Folders'][bright_level]


        img_path = f"{self.path}/{patient}/MRI PNGs/{bright_id}"
        imgs = os.listdir(img_path)
        img_idx = int((idx - row['Start_Index']) % row['Number of Slices']) + 1 # pad zeros?
        img_idx = str(img_idx).zfill(5)
        image = torch.Tensor(np.array(Image.open(f'{img_path}/png_{img_idx}.png'), dtype='int16'))
        
        seg_path = f"{self.path}/{patient}/Seg PNGs"
        label = torch.Tensor(np.array(Image.open(f'{seg_path}/segpng_{img_idx}.png'), dtype='int16'))

        value_to_class = {
            9362: 0, # Background
            18724: 1, # Water
            28086: 2, # Skin
            37449: 3, # Fat
            46811: 4, # FGT?
            56173: 5, # Tumor
            65535: 6 # Clip
        }

        label_classes = torch.zeros_like(label, dtype=torch.long)  # Initialize with zeros
        for value, class_idx in value_to_class.items():
            label_classes[label == value] = class_idx
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label_classes, patient, bright_level