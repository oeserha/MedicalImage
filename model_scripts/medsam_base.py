import os
os.chdir("/home/ra-ugrad/Documents/Haleigh/MedicalImage")

import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from torch.nn import Linear
from torch.nn import Embedding
from segment_anything.modeling.mask_decoder import MLP
import src.settings as settings
from src.medsam_helper import medsam_test_results

def get_base_results(test_loader, device):
    # load model weights
    medsam_model = sam_model_registry['vit_b'](checkpoint=settings.MedSAM_CKPT_PATH)

    # adjust model for task
    medsam_model.mask_decoder.num_mask_tokens = (settings.NUM_CLASSES + 1)
    medsam_model.mask_decoder.num_multimask_outputs = settings.NUM_CLASSES

    medsam_model.image_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size = (35, 35), stride = (3, 3))
    medsam_model.mask_decoder.mask_tokens = Embedding(medsam_model.mask_decoder.num_mask_tokens, 256)
    medsam_model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([MLP(256, 256, 32, 3) for i in range(medsam_model.mask_decoder.num_mask_tokens)])
    medsam_model.mask_decoder.iou_prediction_head.layers[2] = Linear(in_features=256, out_features=medsam_model.mask_decoder.num_mask_tokens, bias=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    medsam_model = medsam_model.to(device)

    # save model
    torch.save(medsam_model.state_dict(), f'./models/base_medsam_model_{settings.DATE}.pth') 

    # get test results
    medsam_model.eval()
    all_base_results = medsam_test_results(test_loader, medsam_model, device)
    grouped_base_results = all_base_results.groupby(["Patient", "Brightness"]).mean()

    return grouped_base_results

