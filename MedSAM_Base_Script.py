# MedSAM Base Testing

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn as nn
from segment_anything import sam_model_registry
from skimage import transform
import torch.nn.functional as F
import matplotlib.pyplot as plt
from boxsdk import OAuth2
from boxsdk import Client
import monai
import shutil
# from io import StringIO

import src.dataloader_box as dl_box
from src.dataloader_box import CancerDataset
import src.medsam_run as medsam_run

# Dataset using Box access
client = dl_box.connection('T3RCQDbC80YZ0fdiXR0wFZel4kJpHPYg')
mri_data = dl_box.get_mri_data(client)
mri_data = dl_box.clean_mri_data(mri_data, client)
train_data, test_data = dl_box.train_test(mri_data)

os.makedirs("/home/haleigh/mri_pics/train")
os.makedirs("/home/haleigh/mri_pics/test")
batch_size = 16
train_dataset = CancerDataset(labels=train_data, client=client)
test_dataset = CancerDataset(labels=test_data, client=client)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

MedSAM_CKPT_PATH = "/home/haleigh/Documents/MedSAM-Base/MedSAM/medsam_vit_b.pth"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model.image_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size = (8, 8), stride = (8, 8), padding = (0, 24))
device = "cuda:0"
medsam_model = medsam_model.to(device)
medsam_model.eval()

# Run MedSAM Base
test_acc = medsam_run.medsam_base_run(test_loader, medsam_model, device)

# Export Results
base_results = pd.DataFrame(test_acc, columns = ['Accuracy', 'Patient', 'Brightness Level'])
base_results.to_csv("MRI_MedSAM_Base.csv", index=False)

shutil.rmtree("/home/haleigh/mri_pics/train")
shutil.rmtree("/home/haleigh/mri_pics/test")