# Medical Image Segmentation-Breast MRI Scans

This project is benchmarking a custom breast MRI dataset on several models, including:
* MedSAM (base model and fine-tuned versions)
* UNet
* DINO
* nn-UNet
* Swin UNETR

The "main.py" file will train each of these models on the dataset and output a results table with the accuracies of each model and the intersection over union ratios for all masks as well as an average across masks. These results are grouped by patient, but may be averaged across patients for total model scoring.

Follow instructions below for setup:

1. Clone the repository: git clone https://github.com/oeserha/MedicalImage.git
2. Create a virtual environment: conda create -n medseg python=3.10 -y and activate it: conda activate medseg
3. Install all required packages: pip install -r requirements.txt
4. Create folder for models (MedicalImage/models)
5. To get MedSAM checkpoint, go to: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link and download the model checkpoint. Place it in the "models" folder
6. Go to src/settings and modify the paths listed to the local locations