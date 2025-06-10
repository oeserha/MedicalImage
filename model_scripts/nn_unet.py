# get dataset structure (json) for nnunet
import subprocess
import sys
import os
import time
import src.settings as settings

from src.data_split_check_store import DataSplitCheckStore
from src.data_format_store import DataFormatStore
from src.data_visualize_pred import DataVisualizer

# download nn-unet (put into MedicalImage)
# create dataset
def create_nnunet_dataset(train_data, test_data):
    dataset_id = 1
    parent_dir = f"{settings.MEDICAL_IMAGE_PATH}/nnUNet_Frame"
    parent_raw_dir = f"{parent_dir}/nnUNet_raw"

    # Create directories for dataset
    if not os.path.exists(parent_raw_dir):
        os.makedirs(parent_raw_dir)
    raw_data_dir = f"{parent_raw_dir}/raw_data/"
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    nnUNet_pred_dir = f"{parent_dir}/nnUNet_predictions/Dataset00{dataset_id}_Breast/raw_predictions"
    if not os.path.exists(nnUNet_pred_dir):
        os.makedirs(nnUNet_pred_dir)
    nnUNet_vis_dir = f"{parent_dir}/nnUNet_predictions/Dataset00{dataset_id}_Breast/visible_predictions"
    if not os.path.exists(nnUNet_vis_dir):
        os.makedirs(nnUNet_vis_dir)
    seg_parent_dir = settings.SEGMENTATIONS_PATH

    train_data_dirs = train_data.patient_id
    test_data_dirs = test_data.patient_id
    # split_percent = 0.75
    folds = [0,1,2,3,4,5,6,7]
    num_epochs = 1
    # Define the label mapping based on the new dataset.json
    label_mapping_format = {
        9362: 0,    # Background
        18724: 1,   # Water
        28086: 2,   # Skin
        37449: 3,   # Fat
        46811: 4,   # FGT
        56173: 5,   # Tumor
        65535: 6    # Clip
    }

    label_mapping_viz = {
        0: 9362,   # Background
        1: 18724,  # Water
        2: 28086,  # Skin
        3: 37449,  # Fat
        4: 46811,  # FGT
        5: 56173,  # Tumor
        6: 65535   # Clip
    }

    # check and store data
    data_split_check_store = DataSplitCheckStore(seg_parent_dir, train_data_dirs, test_data_dirs, raw_data_dir)
    print("Checking and storing data...")
    data_split_check_store.check_and_store_data()
    print("Data checked and stored.")
    brightness_levels = data_split_check_store.brightnesss_levels

    # update labels
    dfs_ts = DataFormatStore(parent_raw_dir, dataset_id, label_mapping_format, brightness_levels, test_set=True)
    dfs_tr = DataFormatStore(parent_raw_dir, dataset_id, label_mapping_format, brightness_levels, test_set=False)
    # Combine and rename data before updating labels
    dfs_ts.combine_rename_data()
    dfs_tr.combine_rename_data()
    print("Updating labels...")
    labelsTs_dir = dfs_ts.update_labels()
    labelsTr_dir = dfs_tr.update_labels()
    print("Labels updated.")

    # Create dataset.json
    print("Creating dataset.json...")
    dfs_tr.create_dataset_json()
    print("Dataset.json created.")

    # Remove the raw_data directory to save space
    print("Removing raw_data directory...")
    dfs_tr.remove_raw_data()
    print("Raw_data directory removed.")

    return brightness_levels

# train model
def train_nnunet(dataset_id, brightness_levels, num_epochs=10):
    parent_dir = f"{settings.MEDICAL_IMAGE_PATH}/nnUNet"
    # run pre-processing
    print("Running pre-processing...")
    os.system(f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
    print("Pre-processing complete.")

    # check if nnUNet_preprocessed folder exists
    preprocessed_dir = f"{parent_dir}/nnUNet_preprocessed/Dataset00{dataset_id}_Breast/nnUNetPlans_2d"
    if not os.path.exists(preprocessed_dir):
        print(f"Preprocessed directory {preprocessed_dir} does not exist. Ensure preprocessing is complete.")
        sys.exit(1)
        
    # run training on two GPUs
    print("Running training...")
    gpu_queue = [0, 1]  # List of available GPUs
    processes = []  # To track running processes

    for brightness in brightness_levels:
        # Wait for an available GPU if all are busy
        while len(processes) >= len(gpu_queue):
            for p in processes:
                if p.poll() is not None:  # Check if the process has finished
                    processes.remove(p)  # Remove finished process from the list

            time.sleep(5)  # Wait before checking again

        # Assign the next available GPU
        gpu = gpu_queue[len(processes) % len(gpu_queue)]
        print(f"Starting training for fold {folds[brightness]} on GPU {gpu}...")
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} nnUNetv2_train {dataset_id} 2d {folds[brightness]} -tr nnUNetTrainer_{num_epochs}epochs --npz"
        processes.append(subprocess.Popen(cmd, shell=True))

    # Wait for all processes to finish
    for p in processes:
        p.wait()

    print("Training complete.")
# save results
def get_nnunet_results():
    pass

def main():
    format_data = True
    training = True
    if training:
        

    if predict:
        # run prediction
        print("Running prediction...")
        for brightness in brightness_levels:
            os.system(f"nnUNetv2_predict -i {parent_raw_dir}/Dataset00{dataset_id}_Breast/imagesTs -o {parent_dir}/nnUNet_predictions/ -d {dataset_id} -c 2d -tr nnUNetTrainer_{num_epochs}epochs -f {brightness}")
        print("Prediction complete.")

        # move predictions to raw_predictions folder
        print("Moving predictions to raw_predictions folder...")
        for image in os.listdir(f"{parent_dir}/nnUNet_predictions/"):
            if image.endswith(".png"):
                src_path = os.path.join(f"{parent_dir}/nnUNet_predictions/", image)
                dst_path = os.path.join(nnUNet_pred_dir, image)
                os.rename(src_path, dst_path)
            else:
                continue

        # visualize predictions
        data_viz = DataVisualizer(label_mapping_viz, dataset_id, nnUNet_pred_dir, nnUNet_vis_dir)
        print("Converting labels back to original format...")
        data_viz.convert_labels_back_to_original_format()
        print("Labels converted back to original format.")

        # delete non-empty nnUNet_preprocessed folder to save space
        print("Deleting nnUNet_preprocessed folder...")
        while os.path.exists(f"{parent_dir}/nnUNet_preprocessed"):
            os.system(f"rm -r {parent_dir}/nnUNet_preprocessed")
        print("nnUNet_preprocessed folder deleted.")

        # delete nnUNet_predictions folder to save space
        print("Deleting nnUNet_predictions folder (non visual predictions)...")
        os.system(f"rm -r {nnUNet_pred_dir}")
        print("nnUNet_predictions folder deleted.")
        print(f"All processes complete. You can now visualize the predictions in {nnUNet_vis_dir}.")

    else:
        print("No data formatting or training/prediction was performed.")

if __name__ == "__main__":
    main()