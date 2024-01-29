import os
import numpy as np
import torch
import torch.nn as nn
from monai.transforms import LoadImaged


def get_files(data_dir):
    """
    Get train/val files from the full OASIS dataset
    """

    data_txt = os.path.join(data_dir, "subjects.txt")

    folders = []

    with open(data_txt, "r") as file:
        for line in file:
            folders.append(str(line).strip())

    # load images and segmentations for training set
    train_files = []
    for folder in folders[:394]:
        train_files.append(
            {
                "image": os.path.join(data_dir, folder, "aligned_norm.nii.gz"),
                "label_4": os.path.join(data_dir, folder, "aligned_seg4.nii.gz"),
                "label_35": os.path.join(data_dir, folder, "aligned_seg35.nii.gz"),
            }
        )

    # load images and segmentations for validation set
    val_files = []
    for i, folder in enumerate(folders[394:-1]):
        next_folder = folders[394:][i + 1]
        val_files.append(
            {
                "fixed_image": os.path.join(data_dir, folder, "aligned_norm.nii.gz"),
                "moving_image": os.path.join(data_dir, next_folder, "aligned_norm.nii.gz"),
                "fixed_label_4": os.path.join(data_dir, folder, "aligned_seg4.nii.gz"),
                "moving_label_4": os.path.join(data_dir, next_folder, "aligned_seg4.nii.gz"),
                "fixed_label_35": os.path.join(data_dir, folder, "aligned_seg35.nii.gz"),
                "moving_label_35": os.path.join(data_dir, next_folder, "aligned_seg35.nii.gz"),
            }
        )

    return train_files, val_files


transform_train = LoadImaged(keys=["image", "label_4", "label_35"], ensure_channel_first=True)
transform_val = LoadImaged(
    keys=["fixed_image", "moving_image", "fixed_label_4", "moving_label_4", "fixed_label_35", "moving_label_35"],
    ensure_channel_first=True,
)
