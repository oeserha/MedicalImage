import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F

class CancerDataset(Dataset):
    def __init__(self, labels, path, train=True, transform=None, target_transform=None):
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.path = path

        self.value_to_class = {
            9362: 0,   # Background
            18724: 1,  # Water
            28086: 2,  # Skin
            37449: 3,  # Fat
            46811: 4,  # FGT
            56174: 5,  # Tumor
            65536: 6   # Clip
        }
    
    def __len__(self):
        return int(sum(self.img_labels['Total Images']))
    
    def __getitem__(self, idx):
        # Get row based on index
        start_idx = self.img_labels.Start_Index
        input_test = self.img_labels.Start_Index.where(start_idx >= idx).first_valid_index()
        if input_test is None:
            input_test = self.img_labels.Start_Index.last_valid_index()
        row = self.img_labels.loc[input_test]
        
        # Calculate brightness level and image index
        patient = row['patient_id']
        bright_level = int((idx - row['Start_Index']) % len(row['Brightness Folders']))
        bright_id = row['Brightness Folders'][bright_level]

        # Load image
        img_path = f"{self.path}{patient}/MRI PNGs/{bright_id}"
        img_idx = int(((idx - row['Start_Index']) % row['Number of Slices'])+1)
        img_idx = f"{img_idx:05d}"
        
        try:
            image = Image.open(f'{img_path}/png_{img_idx}.png').convert("L")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image: {img_path}/png_{img_idx}.png")
            print(f"Error: {str(e)}")
            return None

        # Load and process segmentation mask
        try:
            seg_path = row['PNG segmentation']
            label = np.array(Image.open(f'{seg_path}/segpng_{img_idx}.png'))
            
            label_classes = torch.zeros(label.shape, dtype=torch.long)
            
            for value, class_idx in self.value_to_class.items():
                label_classes[label == value] = class_idx
            
            max_class = label_classes.max().item()
            if max_class >= 7:
                print(f"Warning: Invalid class index found: {max_class}")
                return None

            # the resnet I use needs (224,224) but feel free to change it.
            label_classes = label_classes.unsqueeze(0).unsqueeze(0)  
            label_classes = F.interpolate(
                label_classes.float(),
                size=(224, 224),
                mode='nearest'
            )
            label_classes = label_classes.squeeze(0).squeeze(0).long() 
            
        except Exception as e:
            print(f"Error loading mask: {seg_path}/segpng_{img_idx}.png")
            print(f"Error: {str(e)}")
            return None 

        return image, label_classes, patient, bright_level 
