# Dataloader & Box Integration

from boxsdk import OAuth2
from boxsdk import Client
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Setup connection
def connection(access_token):
    oauth = OAuth2(
    client_id='4ux6qwjvcbp58iau0kgar3t2ceewq47x',
    client_secret='pyYMweGpw6Y7W5WBiWE7bmxymJlDjqoB',
    access_token=access_token,
    )

    # client secret for CCG:

    client = Client(oauth)

    return client

# Access MRI data folder
def get_mri_data(client):

    SHARED_LINK_URL = 'https://app.box.com/file/1433849452284?s=swtqk78ia9uyl2lookmxh26pas75hgrk'
    shared_item = client.get_shared_item(SHARED_LINK_URL)

    with open('FileFromBox.xlsx', 'wb') as open_file:
        client.with_shared_link(SHARED_LINK_URL, None).file(shared_item.id).download_to(open_file)
        open_file.close()

    mri_data = pd.read_excel('FileFromBox.xlsx')
    return mri_data

# Additional Fields for MRI Data
def clean_mri_data(mri_data, client):
    mri_data['Total Images'] = mri_data['Number of Slices'] + mri_data['Number of Brightness Levels']
    mri_data['Start_Index'] = 0
    mri_data['Has MRI'] = ~mri_data['PNG filtered MRI'].isna()
    mri_data['Has Seg'] = ~mri_data['PNG segmentation'].isna()
    folders = []
    for i in range(len(mri_data)):

        brightness_folders = []
        if mri_data['Has MRI'][i]:
            try:
                png = client.get_shared_item(mri_data['PNG filtered MRI'][i])

                items = client.folder(png.id).get_items()
                for item in items:
                    brightness_folders.append(item.id)
                    # brightness_items = client.folder(item.id).get_items()
                    # imgs = []
                    # for img in brightness_items:
                    #     imgs.append(img.id)
                    
                    # brightness_folders.append((item.id, imgs))
            except:
                print(f"{mri_data['MRI/Patient ID'][i]} has error; verify folder addresses")
            
        folders.append(brightness_folders)

    mri_data['Brightness Folders'] = folders

    # filter for only patients with both MRI and Seg PNG files ready
    mri_data = mri_data[(mri_data['Has MRI']) & (mri_data['Has Seg'])]

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
    def __init__(self, labels, client, train = True, transform=None, target_transform=None):
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.client = client

    def __len__(self):
        len = sum(self.img_labels['Total Images'])
        return int(len)

    def __getitem__(self, idx):
        if self.train:
            path = '/home/haleigh/mri_pics/train/'
        else:
            path = '/home/haleigh/mri_pics/test/'
        row = self.img_labels.loc[self.img_labels.Start_Index.where(self.img_labels.Start_Index >= idx).first_valid_index()]
        
        patient = row['MRI/Patient ID']
        bright_level = int((idx - row['Start_Index']) % len(row['Brightness Folders']))
        bright_id = row['Brightness Folders'][bright_level]

        brightness_items = self.client.folder(bright_id).get_items()
        imgs = []
        for img in brightness_items:
            imgs.append(img.id)

        img_idx = int((idx - row['Start_Index']) % row['Number of Slices'])
        with open(f'{path}png_{idx}.png', 'wb') as open_pic:
            self.client.file(imgs[img_idx]).download_to(open_pic)
            open_pic.close()

        image = torch.Tensor(np.array(Image.open(f'{path}png_{idx}.png'), dtype='int16'))
        
        seg_folder = self.client.get_shared_item(row['PNG segmentation']).get_items()
        seg_ids = []
        for item in seg_folder:
            seg_ids.append(item.id)

        with open(f'{path}seg_{idx}.png', 'wb') as open_pic:
            self.client.file(seg_ids[img_idx]).download_to(open_pic)
            open_pic.close()

        label = torch.Tensor(np.array(Image.open(f'{path}seg_{idx}.png'), dtype='int16'))

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