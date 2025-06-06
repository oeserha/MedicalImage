import os
os.chdir("/home/ra-ugrad/Documents/Haleigh/MedicalImage")

import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from torch.nn import Linear
from torch.nn import Embedding
from segment_anything.modeling.mask_decoder import MLP
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from monai.losses import DiceLoss

from src.medsam_helper import medsam_test_results
from src.medsam_helper import MedSAM
from src.utils import get_class_weights
import src.settings as settings

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_tuned_model(train_loader, num_epochs=10):
    # load model weights
    medsam_model = sam_model_registry['vit_b'](checkpoint=settings.MedSAM_CKPT_PATH)

    # adjust model for task
    medsam_model.mask_decoder.num_mask_tokens = (settings.NUM_CLASSES + 1)
    medsam_model.mask_decoder.num_multimask_outputs = settings.NUM_CLASSES

    medsam_model.image_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size = (35, 35), stride = (3, 3))
    medsam_model.mask_decoder.mask_tokens = Embedding(medsam_model.mask_decoder.num_mask_tokens, 256)
    medsam_model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([MLP(256, 256, 32, 3) for i in range(medsam_model.mask_decoder.num_mask_tokens)])
    medsam_model.mask_decoder.iou_prediction_head.layers[2] = Linear(in_features=256, out_features=medsam_model.mask_decoder.num_mask_tokens, bias=True)
    
    tune_model = MedSAM(
        image_encoder=medsam_model.image_encoder,
        mask_decoder=medsam_model.mask_decoder,
        prompt_encoder=medsam_model.prompt_encoder,
    ).to(device)
    tune_model = tune_model.to(device)

    # train model
    tune_model.train()
    img_mask_encdec_params = list(tune_model.image_encoder.parameters()) + list(
        tune_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=1e-3, weight_decay=.01
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

    num_epochs = num_epochs
    losses = []
    accs = []
        
    for epoch in range(num_epochs):
        for step, (img, seg, patient, b_level) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            seg = seg.to(device)

            optimizer.zero_grad()
            B, H, W = img.size()

            img_3c = img.repeat(3, 1, 1, 1).view(B, 3, H, W).to(device)
            boxes_np = torch.Tensor(np.array([[0, 0, W, H]])).detach().cpu().numpy()
            img, seg = img.to(device), seg.to(device)

            logits = tune_model(img_3c, boxes_np)
            pred = torch.argmax(logits, dim=1)

            acc = (torch.tensor(pred).clone().detach() == seg).float().mean()
            accs.append(acc.item())

            loss = seg_loss(logits, seg.unsqueeze(dim=1)) + ce_loss(logits, seg)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #TODO: add pbar

    # save model
    torch.save(tune_model.state_dict(), f'./models/tuned_medsam_model_{settings.DATE}.pth')
        
    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.savefig(f"./figs/tuned_losses_{settings.DATE}.png")

    plt.plot(accs)
    plt.title("Pixel Accuracy")
    plt.xlabel("Train Step")
    plt.ylabel("Accuracy (All Masks)")
    plt.savefig(f"./figs/tuned_losses_{settings.DATE}.png")

def get_tuned_results(tune_model_path, test_loader, device):
    # get test results
    medsam_model = sam_model_registry['vit_b'](checkpoint=settings.MedSAM_CKPT_PATH)
    medsam_model.mask_decoder.num_mask_tokens = 8
    medsam_model.mask_decoder.num_multimask_outputs = 7
    medsam_model.image_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size = (35, 35), stride = (3, 3))
    medsam_model.mask_decoder.mask_tokens = Embedding(medsam_model.mask_decoder.num_mask_tokens, 256)
    medsam_model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([MLP(256, 256, 32, 3) for i in range(medsam_model.mask_decoder.num_mask_tokens)])
    medsam_model.mask_decoder.iou_prediction_head.layers[2] = Linear(in_features=256, out_features=medsam_model.mask_decoder.num_mask_tokens, bias=True)
    medsam_model.load_state_dict(torch.load(tune_model_path))
    medsam_model.eval()
    all_tuned_results = medsam_test_results(test_loader, medsam_model, device)
    grouped_tuned_results = all_tuned_results.groupby(["Patient", "Brightness"]).mean()

    return grouped_tuned_results