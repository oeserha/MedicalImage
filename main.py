# TODO: run scripts (base, finetuned, CNN, DINO, nn-Unet, SWIN)
from model_scripts.medsam_base import get_base_results
import src.dataloader_local as dl_local
from src.dataloader_local import CancerDataset
from torch.utils.data import DataLoader
import pandas as pd

def main():

    # get all mris for training/testing
    mri_data = dl_local.get_mri_data()
    mri_data = dl_local.clean_mri_data(mri_data)

    # set train/test data
    train_data, test_data = dl_local.train_test(mri_data)
    batch_size = 8

    train_dataset = CancerDataset(labels=train_data, path="/home/ra-ugrad/Documents/Segmentations/")
    test_dataset = CancerDataset(labels=test_data, path="/home/ra-ugrad/Documents/Segmentations/")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # get results for each model
    base_results = get_base_results(test_loader)
    base_results['Model'] = "Base"

    # fine-tuned
    # cnn
    # nn-unet

    all_results = base_results


if __name__ == "__main__":
    main()