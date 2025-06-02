# All universal variables
from datetime import datetime

MEDICAL_IMAGE_PATH = "/home/ra-ugrad/Documents/Haleigh/MedicalImage"
SEGMENTATIONS_PATH = "/home/ra-ugrad/Documents/Segmentations"
SEGMENTATIONS_LIST = "seg_list_test.xlsx"
MedSAM_CKPT_PATH = "./models/medsam_vit_b.pth"

#TODO: add to all models
IMAGE_SIZE = 224
NUM_CLASSES = 7
BATCH_SIZE = 2
NUM_EPOCHS = 1
DATE = datetime.now().strftime("%m-%d-%Y") #TODO: add to all models for versioning