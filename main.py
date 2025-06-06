from model_scripts.medsam_base import get_base_results
from model_scripts.medsam_tuned import get_tuned_results
from model_scripts.medsam_tuned import train_tuned_model
from model_scripts.unet import train_unet
from model_scripts.unet import get_unet_results
from model_scripts.dino import train_dino
from model_scripts.dino import get_dino_results
from model_scripts.nn_unet import create_nnunet_dataset
from model_scripts.nn_unet import train_nnunet
from model_scripts.nn_unet import get_nnunet_results

import src.data_helper as dl_local
from src.data_helper import CancerDataset
import src.settings as settings
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from datetime import datetime
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # get all mris for training/testing
    mri_data = dl_local.get_mri_data()
    mri_data = dl_local.clean_mri_data(mri_data)

    # set train/test data
    train_data, test_data = dl_local.train_test(mri_data)
    batch_size = settings.BATCH_SIZE

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        # This is the mean and std of ImageNet averaged across RGB channels
        transforms.Normalize(mean=[0.449], std=[0.226])
    ])

    #TODO: change to pass data into models so we can have custom transforms
    train_dataset = CancerDataset(labels=train_data, path=settings.SEGMENTATIONS_PATH)#, transform=transform)
    test_dataset = CancerDataset(labels=test_data, path=settings.SEGMENTATIONS_PATH)#, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    # get results for each model
    # base medsam
    print("Getting base MedSAM results...")
    base_results = get_base_results(test_loader, device)
    base_results['Model'] = "Base" 
    print("Base MedSAM complete.")

    # fine-tuned medsam
    print("Training MedSAM...")
    train_tuned_model(train_loader, num_epochs=1) 
    print("Getting MedSAM results...")
    tuned_results = get_tuned_results("./models/tuned_medsam_model_06-04-2025.pth", test_loader, device) 
    tuned_results['Model'] = "Tuned"
    print("MedSAM complete.")

    # unet
    print("Training UNet...")
    unet_model = train_unet(train_loader, device, num_epochs=1)
    print("Getting UNet results...")
    unet_results = get_unet_results(unet_model, test_loader, device)
    unet_results['Model'] = "CNN"
    print("UNet complete.")

    # nn-unet
    print("Creating nnUNet dataset...")
    brightness_levels = create_nnunet_dataset(train_data, test_data)
    print("Training nnUNet...")
    train_nnunet(1, brightness_levels, 1)
    print("Getting nnUNet results...")
    nnUNet_results = get_nnunet_results()
    nnUNet_results['Model'] = "nnUNet"
    print("nnUNet complete.")

    # DINO
    print("Training DINO...")
    dino_model = train_dino(train_loader, device, num_epochs=1)
    print("Getting DINO results...")
    dino_results = get_dino_results(dino_model, test_loader, device)
    dino_results['Model'] = "DINO"
    print("DINO complete.")

    # swin unetr
    #TODO: write script for swin unetr

    all_results = pd.concat([base_results, tuned_results, unet_results, dino_results])

    # output results
    pd.to_csv(all_results, f"./results/results_{settings.DATE}.csv")

    print("All models trained and results posted.")


if __name__ == "__main__":
    main()