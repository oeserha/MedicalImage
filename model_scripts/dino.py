import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from monai.losses import DiceLoss
from tqdm import tqdm

from src.data_helper import CancerDataset
from src.utils import calculate_iou
from src.utils import get_class_weights
import src.settings as settings

# Define the segmentation model with DINOv2 backbone
class DINOv2Segmentation(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize DINOv2 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.feature_dim = 384  # DINOv2-small feature dimension
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Decoder network with skip connections
        self.decoder = nn.Sequential(
            # First block
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final 1x1 conv for classification
            nn.Conv2d(64, 7, kernel_size=1)
        )
        
    def forward(self, x):
        # Get features from DINOv2 backbone
        features_dict = self.backbone.forward_features(x)
        
        # Extract patch tokens from the dictionary
        features = features_dict['x_norm_patchtokens']  # Shape: [B, N, D]
        B = x.shape[0]
        
        # Reshape features from [B, N, D] to [B, D, H, W]
        # N = H * W = 16 * 16 = 256 for 224x224 input
        features = features.permute(0, 2, 1)  # [B, D, N]
        features = features.reshape(B, -1, 16, 16)  # [B, D, H, W]
        
        # Apply decoder
        x = self.decoder(features)
        
        # Upsample to input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x

def set_data_loaders(train_data, test_data):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    print("Creating datasets...")
    train_dataset = CancerDataset(
        labels=train_data,
        path=settings.SEGMENTATIONS_PATH,
        transform=transform
    )

    test_dataset = CancerDataset(
        labels=test_data,
        path=settings.SEGMENTATIONS_PATH,
        transform=transform,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def train_dino(train_loader, device, num_epochs=10):
    print("Creating model...")
    model = DINOv2Segmentation()
    model = model.to(device)

    class_weights = get_class_weights()
    class_weights = class_weights.to(device)

    # Loss function with class weights to combat class imbalance
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    seg_loss = DiceLoss(
        to_onehot_y=True, 
        softmax=True,
        include_background=True,
        batch=True,
        weight=class_weights
    )
    # seg_loss = DiceLoss(softmax=True, squared_pred=True, reduction="mean") 
    # #TODO: check other params for importance

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("Starting training...")
    model.train()
    losses = []
    accs = []

    for epoch in range(num_epochs):
        for step, (img, seg, patient, b_level) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            seg = seg.to(device)

            #TODO: when switching to using custom dataset w/ transforms, delete this
            B, H, W = img.size()
            img_3c = img.repeat(3, 1, 1, 1).view(B, 3, H, W)
            mask = seg.unsqueeze(dim=1)

            outputs = model(img_3c)
            outputs = outputs.view(-1, settings.NUM_CLASSES, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
            pred = torch.argmax(outputs, dim=1)
            
            loss = seg_loss(outputs, mask) + ce_loss(outputs, seg)
            acc = (pred == seg).float().mean()
            accs.append(acc.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.savefig(f"./figs/dino_train_loss_{settings.DATE}.png")

    plt.plot(accs)
    plt.title("Pixel Accuracy")
    plt.xlabel("Train Step")
    plt.ylabel("Accuracy (All Masks)")
    plt.savefig(f"./figs/dino_train_acc_{settings.DATE}.png")

    print("Saving model...")
    torch.save(model.state_dict(), f'./models/dinov2_model_{settings.DATE}.pth')

    return model
        

def get_dino_results(model, test_loader, device):
    results = pd.DataFrame(columns = ["Patient", "Brightness", "Accuracy", "IoU_0", "IoU_1", "IoU_2"
                                      "IoU_3", "IoU_4", "IoU_5", "IoU_6", "IoU_mean"])

    class_weights = get_class_weights()
    class_weights = class_weights.to(device)

    model.eval()
    with torch.no_grad():
        for img, seg, patient, b_level in test_loader:
            img = img.to(device)
            seg = seg.to(device)

            B, H, W = img.size()
            img_3c = img.repeat(3, 1, 1, 1).view(B, 3, H, W)
            
            outputs = model(img_3c)
            outputs = outputs.view(-1, settings.NUM_CLASSES, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
            pred = torch.argmax(outputs, dim=1)
            
            acc = list((pred == seg).float().mean(dim =(1, 2)).cpu().numpy())
            mean_iou, class_iou = calculate_iou(outputs, seg, settings.NUM_CLASSES)

            results = pd.concat([results, pd.DataFrame({"Patient": patient, 
                                                        "Brightness": b_level, 
                                                        "Accuracy": acc,
                                                        "IoU_0": class_iou[0],
                                                        "IoU_1": class_iou[1],
                                                        "IoU_2": class_iou[2],
                                                        "IoU_3": class_iou[3], 
                                                        "IoU_4": class_iou[4],
                                                        "IoU_5": class_iou[5],
                                                        "IoU_6": class_iou[6],
                                                        "IoU_mean": mean_iou,
                                                        })]) 
            grouped_dino_results = results.groupby(["Patient", "Brightness"]).mean()

            return grouped_dino_results