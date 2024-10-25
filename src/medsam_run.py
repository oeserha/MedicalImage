import numpy as np
import torch
import matplotlib.pyplot as plt
from src.medsam_helper import show_box, show_mask, medsam_inference

def medsam_base_run(test_loader, medsam_model, device):
    test_acc = []
    i = 0
    for x, y, patient, b_level in test_loader:
        i = 0
        img = x
        B, H, W = img.size()
        img_3c = img.repeat(3, 1, 1, 1).view(B, 3, H, W).to(device)

        box_np = np.array([[0,0, W, H]]).to(device)

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_3c) # (1, 256, 64, 64)
        
        medsam_seg = medsam_inference(medsam_model, image_embedding, box_np, H, W)

        acc = (torch.tensor(medsam_seg) == y).float().mean()
        test_acc.append((acc, patient, b_level))

        if i == 0:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img, cmap="gray")
            show_box(box_np[0], ax[0])
            ax[0].set_title("Input Image and Bounding Box")
            ax[1].imshow(img, cmap="gray")
            show_mask(medsam_seg, ax[1])
            show_box(box_np[0], ax[1])
            ax[1].set_title(f"MedSAM Segmentation w/ Accuracy: {acc}")
            plt.show()
        
        i += 1

    return test_acc