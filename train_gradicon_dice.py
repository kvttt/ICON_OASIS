# python imports
import os
import glob
import tempfile
import time
import warnings
from pprint import pprint
import shutil

# data science imports
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

# MONAI imports
from monai.apps import download_and_extract
from monai.data import Dataset, CacheDataset, DataLoader
from monai.losses import DiffusionLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import VoxelMorphUNet
from monai.networks.utils import one_hot
from monai.utils import set_determinism, first
from monai.visualize.utils import blend_images
from monai.config import print_config
from monai.transforms import LoadImaged

# Local imports
from script import download_oasis
from src.losses import GradientICONSparse, GradientICON
from src.network_wrappers import FunctionFromVectorField
from src.utils import get_files, transform_train, transform_val
from src.monai_wrapper import make_ddf_using_icon_module, ConcatInputs, FirstChannelInputs

set_determinism(seed=0)
torch.backends.cudnn.benchmark = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore")

# device, optimizer, epoch and batch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
lr = 1e-4
weight_decay = 1e-5
max_epochs = 50

# use mixed precision feature of GPUs for faster training
amp_enabled = True

# loss weights (set to zero to disable loss term)
lam_sim = 1e0  # MSE (image similarity)
# lam_smooth = 1e-2  # Bending loss (smoothness)
lam_dice = 5e-2  # Dice (auxiliary)

# whether to use coarse (4-label) or fine (35-label) labels for training
use_coarse_labels = True

# write model and tensorboard logs?
do_save = True
dir_save = os.path.join(os.getcwd(), "models", "voxelmorph_GradICON_002")  # change this for different experiments
if do_save and not os.path.exists(dir_save):
    os.makedirs(dir_save)

# data directory
data_dir = './OASIS'

# define dataset and dataloader
train_files, val_files = get_files(data_dir)
train_ds = Dataset(data=train_files, transform=transform_train)
val_ds = Dataset(data=val_files, transform=transform_val)
train_loader = DataLoader(train_ds, batch_size=2 * batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)

# model
inner_net = FunctionFromVectorField(
    ConcatInputs(
        VoxelMorphUNet(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=32,
            channels=(16, 32, 32, 32, 32, 32),
            final_conv_channels=(16, 16)
        )
    )
)
# initialize the model with identity map
nn.init.zeros_(inner_net.net.net.net[1].final_conv_out.conv.weight)
nn.init.zeros_(inner_net.net.net.net[1].final_conv_out.conv.bias)

net = GradientICON(
    inner_net,
    FirstChannelInputs(MSELoss()),
    0.02
)

net.assign_identity_map((1, 1, 160, 192, 224))
net.to(device)

warp_layer = Warp().to(device)

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# metrics
dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)


# define forward pass
def forward(fixed_image, moving_image, moving_label, net, warp_layer, num_classes):
    """
    Model forward pass:
        - predict DDF,
        - convert moving label to one-hot format, and
        - warp one-hot encoded moving label using predicted DDF
    """

    # predict DDF and warp moving image using predicted DDF
    ddf_image, iconloss = make_ddf_using_icon_module(
        net,
        moving_image,
        fixed_image
    )  # return a monai compatible ddf and ICONLoss object
    all_loss = iconloss.all_loss
    similarity_loss = iconloss.similarity_loss

    # warp moving label
    # num_classes + 1 to include background as a channel
    moving_label_one_hot = one_hot(moving_label, num_classes=num_classes + 1)
    pred_label_one_hot = warp_layer(moving_label_one_hot, ddf_image)

    return all_loss, similarity_loss, pred_label_one_hot


# define loss function
def loss_fun(
        all_loss,
        fixed_label,
        pred_label_one_hot,
        lam_sim,
        lam_dice,
):
    """
    Custom multi-target loss:
        - Image similarity: MSELoss
        - Deformation smoothness: BendingEnergyLoss
        - Auxiliary loss: DiceLoss
    """
    # Instantiate where necessary
    if lam_dice > 0:
        # we exclude the first channel (i.e., background) when calculating dice
        label_loss = DiceLoss(include_background=False)

    num_classes = 4 if use_coarse_labels else 35

    # Compute loss components
    dice = label_loss(pred_label_one_hot, one_hot(fixed_label, num_classes=num_classes + 1)) if lam_dice > 0 else 0

    # Weighted combination:
    return lam_sim * all_loss + lam_dice * dice


def train():
    # Automatic mixed precision (AMP) for faster training
    amp_enabled = False
    scaler = torch.cuda.amp.GradScaler()

    # Tensorboard
    if do_save:
        writer = SummaryWriter(log_dir=dir_save)

    # Start torch training loop
    val_interval = 1
    best_eval_dice = 0
    log_train_loss = []
    log_train_mse_loss = []
    log_val_dice = []
    pth_best_dice, pth_latest = "", ""

    for epoch in range(max_epochs):
        # ==============================================
        # Train
        # ==============================================
        net.train()

        epoch_loss, n_steps = 0, 0
        epoch_mse_loss = 0
        t0_train = time.time()
        for batch_data in tqdm(train_loader):
            # for batch_data in tqdm(train_loader):
            # Get data: manually slicing along the batch dimension to obtain the fixed and moving images
            fixed_image = batch_data["image"][0:1, ...].to(device)
            moving_image = batch_data["image"][1:, ...].to(device)
            if use_coarse_labels:
                fixed_label = batch_data["label_4"][0:1, ...].to(device)
                moving_label = batch_data["label_4"][1:, ...].to(device)
            else:
                fixed_label = batch_data["label_35"][0:1, ...].to(device)
                moving_label = batch_data["label_35"][1:, ...].to(device)
            n_steps += 1

            # Forward pass and loss
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast(enabled=amp_enabled):
            all_loss, similarity_loss, pred_label_one_hot = forward(
                fixed_image, moving_image, moving_label, net, warp_layer, num_classes=4
            )
            loss = loss_fun(
                all_loss,
                fixed_label,
                pred_label_one_hot,
                lam_sim,
                lam_dice,
            )
            # Optimise
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            epoch_loss += loss.item()
            epoch_mse_loss += similarity_loss.item()

        # Scheduler step
        lr_scheduler.step()
        # Loss
        epoch_loss /= n_steps
        epoch_mse_loss /= n_steps
        log_train_loss.append(epoch_loss)
        log_train_mse_loss.append(epoch_mse_loss)
        if do_save:
            writer.add_scalar("train/loss", epoch_loss, epoch)
            writer.add_scalar("train/mse_loss", epoch_mse_loss, epoch)
        print(f"{epoch + 1} | loss = {epoch_loss:.6f} " f"elapsed time: {time.time() - t0_train:.2f} sec.")
        # ==============================================
        # Eval
        # ==============================================
        if (epoch + 1) % val_interval == 0:
            net.eval()

            n_steps = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    # Get data
                    fixed_image = batch_data["fixed_image"].to(device)
                    moving_image = batch_data["moving_image"].to(device)
                    fixed_label_4 = batch_data["fixed_label_4"].to(device)
                    moving_label_4 = batch_data["moving_label_4"].to(device)
                    # fixed_label_35 = batch_data["fixed_label_35"].to(device)
                    # moving_label_35 = batch_data["moving_label_35"].to(device)
                    n_steps += 1
                    # Infer
                    # with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # 	all_loss, pred_label_one_hot = forward(
                    # 		fixed_image, moving_image, moving_label_4, net, warp_layer, num_classes=4
                    # 	)
                    ddf, _ = make_ddf_using_icon_module(
                        net,
                        moving_image,
                        fixed_image
                    )
                    moving_label_one_hot = one_hot(moving_label_4, num_classes=5)
                    pred_label_one_hot = warp_layer(moving_label_one_hot, ddf)

                    # Dice
                    dice_metric_before(y_pred=moving_label_4, y=fixed_label_4)
                    dice_metric_after(y_pred=pred_label_one_hot.argmax(dim=1, keepdim=True), y=fixed_label_4)

            # Dice
            dice_before = dice_metric_before.aggregate().item()
            dice_metric_before.reset()
            dice_after = dice_metric_after.aggregate().item()
            dice_metric_after.reset()
            if do_save:
                writer.add_scalar("val/dice", dice_after, epoch)
            log_val_dice.append(dice_after)
            print(f"{epoch + 1} | dice_before = {dice_before:.3f}, dice_after = {dice_after:.3f}")

            if dice_after > best_eval_dice:
                best_eval_dice = dice_after
                if do_save:
                    # Save best model based on Dice
                    if pth_best_dice != "":
                        os.remove(os.path.join(dir_save, pth_best_dice))
                    pth_best_dice = f"voxelmorph_loss_best_dice_{epoch + 1}_{best_eval_dice:.3f}.pth"
                    torch.save(net.state_dict(), os.path.join(dir_save, pth_best_dice))
                    print(f"{epoch + 1} | Saving best Dice model: {pth_best_dice}")

        if do_save:
            # Save latest model
            if pth_latest != "":
                os.remove(os.path.join(dir_save, pth_latest))
            pth_latest = "voxelmorph_latest.pth"
            torch.save(net.state_dict(), os.path.join(dir_save, pth_latest))


if __name__ == "__main__":
    print_config()
    train()
