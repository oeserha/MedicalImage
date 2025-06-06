import torch
import numpy as np
from segmentation_models_pytorch import Unet
from monai.losses import DiceLoss
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from src.utils import calculate_iou
from src.utils import get_class_weights
import src.settings as settings

warnings.filterwarnings("ignore", category=UserWarning)

def train_unet(train_loader, device, num_epochs=10):
    print("Creating model...")
    model = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=7)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=.01
    )

    class_weights = get_class_weights().to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    seg_loss = DiceLoss(
        to_onehot_y=True, 
        softmax=True,
        include_background=True,
        batch=True,
        weight=class_weights
    )

    losses = []
    accs = []

    for epoch in range(num_epochs):
        model.train()
        
        for step, (img, seg, patient, b_level) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            seg = seg.to(device)

            img = img.unsqueeze(dim=1)
            
            outputs = model(img)
            pred = torch.argmax(outputs, dim=1)

            acc = (pred.detach() == seg).float().mean()
            accs.append(acc.item())

            loss = seg_loss(outputs, seg.unsqueeze(dim=1)) + ce_loss(outputs, seg)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #TODO: add pbar

    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.savefig(f"./figs/unet_losses_{settings.DATE}.png")

    plt.plot(accs)
    plt.title("Pixel Accuracy")
    plt.xlabel("Train Step")
    plt.ylabel("Accuracy (All Masks)")
    plt.savefig(f"./figs/unet_losses_{settings.DATE}.png")

    torch.save(model.state_dict(), f'./models/unet_model_{settings.DATE}.pth')

    return model #TODO: remove model return (take it from saved model)

def get_unet_results(model, test_loader, device):
    results = pd.DataFrame(columns = ["Patient", "Brightness", "Accuracy", "IoU_0", "IoU_1", "IoU_2"
                                      "IoU_3", "IoU_4", "IoU_5", "IoU_6", "IoU_mean"])

    model.eval()
    with torch.no_grad():
        for img, seg, patient, b_level in test_loader:
            img = img.to(device)
            seg = seg.to(device)

            img = img.unsqueeze(dim=1)

            outputs = model(img)
            pred = torch.argmax(outputs, dim=1)

            acc = list((pred == seg).float().mean(dim =(1, 2)).cpu().numpy())
            mean_iou, class_iou = calculate_iou(outputs.cpu(), seg.cpu(), 7)

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

    grouped_unet_results = results.groupby(["Patient", "Brightness"]).mean()
    return grouped_unet_results


