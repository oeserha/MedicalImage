import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    with torch.no_grad():
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
            )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu()
        pred = torch.argmax(low_res_pred, dim=1)
        return pred
    
def medsam_test_results(test_loader, medsam_model, device):
    results = pd.DataFrame(columns = ["Patient", "Brightness", "Accuracy"])
    for x, y, patient, b_level in test_loader:
        img = x
        B, H, W = img.size()
        img_3c = img.repeat(3, 1, 1, 1).view(B, 3, H, W).to(device)

        box_np = torch.Tensor(np.array([[0, 0, W, H]])).to(device)

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_3c) # (1, 256, 64, 64)
        
        medsam_seg = medsam_inference(medsam_model, image_embedding, box_np, H, W)

        acc = (torch.tensor(medsam_seg) == y).float().mean(dim=(1, 2))
        results = pd.concat([results, pd.DataFrame({"Patient": patient, "Brightness": b_level, "Accuracy": acc})])

    return results

def medsam_training(train_loader, medsam_model, device):
    # set objective fcn
    for x, y, patient, b_level in train_loader:
        img = x
        B, H, W = img.size()
        img_3c = img.repeat(3, 1, 1, 1).view(B, 3, H, W).to(device)

        box_np = torch.Tensor(np.array([[0, 0, W, H]])).to(device)

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_3c) # (1, 256, 64, 64)
        
        medsam_seg = medsam_inference(medsam_model, image_embedding, box_np, H, W)
        # get loss
    pass
