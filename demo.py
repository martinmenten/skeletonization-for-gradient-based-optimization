import imageio
import numpy as np
import torch

from skeletonize import Skeletonize


# Two-dimensional example from the DRIVE dataset
img = imageio.imread('data/image_drive.png') / 255.
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

skeletonization_module = Skeletonize(probabilistic=False, simple_point_detection='Boolean')
skeleton = skeletonization_module(img)

skeleton = skeleton.numpy().squeeze() * 255
imageio.imwrite('data/skeleton_drive.png', skeleton.astype(np.uint8))


# Same example with added uniform noise to demonstrate skeletonization of a non-binary input
img = imageio.imread('data/image_drive_added_noise.png') / 255.
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

skeletonization_module = Skeletonize(probabilistic=True, beta=0.33, tau=1.0, simple_point_detection='Boolean')
skeleton = skeletonization_module(img)

skeleton = skeleton.numpy().squeeze() * 255
imageio.imwrite('data/skeleton_drive_added_noise.png', skeleton.astype(np.uint8))


# Application of the skeletonization module multiple times (as done commonly in gradient-based optimization)
# so that the output converges towards the true skeleton
img = imageio.imread('data/image_drive_added_noise.png') / 255.
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

skeletonization_module = Skeletonize(probabilistic=True, beta=0.33, tau=1.0, simple_point_detection='Boolean')
skeleton_stack = np.zeros_like(img.squeeze())
for step in range(20):
    skeleton_stack = skeleton_stack + skeletonization_module(img).numpy().squeeze()
skeleton = (skeleton_stack / 20).round()

skeleton = skeleton * 255
imageio.imwrite('data/skeleton_drive_added_noise_20steps.png', skeleton.astype(np.uint8))


# Three-dimensional example from the VESSAP dataset
img = np.load('data/image_vessap.npy')
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

skeletonization_module = Skeletonize(probabilistic=False, simple_point_detection='Boolean', num_iter=10)
skeleton = skeletonization_module(img)

skeleton = skeleton.numpy().squeeze()
np.save('data/skeleton_vessap.npy', skeleton)
