# Trying GradICON on Learn2Reg 2021 Task 3 (OASIS)

## Note
In this training script, MSE is used as the similarity metric. 
The original paper suggests setting $\lambda$ to 0.2 when MSE is used
(see Table 2 caption) in the [original paper](https://arxiv.org/pdf/2206.05897.pdf).
In this training script, I set $\lambda = 0.02$, which empirically gives better result.
This is probably because the original paper set $\lambda = 0.2$ for a 2D task, 
but here with OASIS we are dealing with 3D images.

In addition, I added Dice loss as an auxiliary loss.

The training script requires the latest version of MONAI. Install by:
```bash
pip install -q monai-weekly
```

Run the following in Python to check MONAI version:
```python
from monai.config import print_config
print_config()
```

## Acknowledgement
This training script is based on [this MONAI tutorial](https://github.com/Project-MONAI/tutorials/blob/main/3d_registration/learn2reg_oasis_unpaired_brain_mr.ipynb).

Also see [ICON](https://github.com/uncbiag/ICON) for GradICON.

## Further steps
- [ ] Try other similarity metrics, e.g., LNCC.
- [ ] Train with AMP (automatic mixed precision).
- [ ] Only very coarse hyperparameter tuning was done. Therefore,
  - [ ] Tune hyperparameters more carefully.
  - [ ] Try hypernetwork-based, e.g., HyperMorph-like, design.
- [ ] The UNet architecture used here is the Vanilla UNet used in VoxelMorph. Therefore,
  - [ ] Try other architectures, e.g., ViT will be a good candidate (implemented in MONAI). 