import torch
import torchvision
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# Load your data (replace this path with the actual path to your .pt file)
data = torch.load('/home/shuheng/MDC_MMD/raid/results/cifar10/conv3in_grad_mse_nd2000_cut_niter2000_factor2_lr0.005_mix_ipc10_wo_mse/data_2000.pt')

# Separate the images and labels
images, labels = data

# Check if 'images' is already a tensor
if isinstance(images, torch.Tensor):
    images_tensor = images  # If images is already a tensor, use it directly
else:
    images_tensor = torch.stack(images)  # Otherwise, stack them into a single tensor

# Create a grid from the images, assuming a 10-class setup for CIFAR-10
grid = make_grid(images_tensor, nrow=10)  # Adjust `nrow` if needed for your class setup

# Save the image grid to a PNG file
save_image(grid, '/home/shuheng/MDC_MMD/raid/results/cifar10/conv3in_grad_mse_nd2000_cut_niter2000_factor2_lr0.005_mix_ipc10_wommd/pure_mmd.png')

# Alternatively, display the grid using matplotlib
plt.imshow(grid.permute(1, 2, 0))  # Reorder dimensions from [C, H, W] to [H, W, C]
plt.axis('off')
plt.show()
