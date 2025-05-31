import os
os.chdir("/home/ra-ugrad/Documents/Haleigh/MedicalImage")

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from torch.nn import Linear
from torch.nn import Embedding
from segment_anything.modeling.mask_decoder import MLP

import src.dataloader_local as dl_local
from src.dataloader_ryan import CancerDataset # TODO: check why it's not working from local
from src.medsam_helper import medsam_test_results
from src.medsam_helper import MedSAM
from src.medsam_helper import medsam_training

def get_tuned_results(train_loader, test_loader):
    # load model weights
    MedSAM_CKPT_PATH = "./models/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)

    # adjust model for task
    medsam_model.mask_decoder.num_mask_tokens = 8
    medsam_model.mask_decoder.num_multimask_outputs = 7

    medsam_model.image_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size = (35, 35), stride = (3, 3))
    medsam_model.mask_decoder.mask_tokens = Embedding(medsam_model.mask_decoder.num_mask_tokens, 256)
    medsam_model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([MLP(256, 256, 32, 3) for i in range(medsam_model.mask_decoder.num_mask_tokens)])
    medsam_model.mask_decoder.iou_prediction_head.layers[2] = Linear(in_features=256, out_features=medsam_model.mask_decoder.num_mask_tokens, bias=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tune_model = MedSAM(
        image_encoder=medsam_model.image_encoder,
        mask_decoder=medsam_model.mask_decoder,
        prompt_encoder=medsam_model.prompt_encoder,
    ).to(device)
    tune_model = tune_model.to(device)

    # train model
    tune_model = medsam_training(train_loader, tune_model, device)

    # save model
    torch.save(tune_model.state_dict(), './models/tuned_medsam_model.pth')

    # get test results
    tune_model.eval()
    all_tuned_results = medsam_test_results(test_loader, tune_model, device)
    grouped_tuned_results = all_tuned_results.groupby(["Patient", "Brightness"]).mean()

    return grouped_tuned_results