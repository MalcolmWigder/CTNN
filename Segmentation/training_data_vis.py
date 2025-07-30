#!/usr/bin/env python3
"""
visualize_tensorstore_color.py

Edit TENSORSTORE_PATH and INDICES below, then run:
    python visualize_tensorstore_color.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# ←── Set these
TENSORSTORE_PATH = "tensorstore.pt"
INDICES = [1000, 3000, 5000]  # list any integer indices you want to visualize

def visualize_colored_masks(path, indices):
    data = torch.load(path, map_location="cpu")
    images = data["images"]    # shape (N,1,H,W)
    masks  = data["masks"]     # list of length N, each a tensor

    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(6*2, 4*n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row, idx in enumerate(indices):
        # Radar image
        img = images[idx].cpu().numpy().squeeze()  # (H,W)
        
        # Instance masks: ensure shape is (num_instances, H, W)
        mask = masks[idx].cpu().numpy()
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]
        
        H, W = mask.shape[1:]
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Assign a random color to each cloud instance
        for inst in range(mask.shape[0]):
            # generate a random RGB triplet
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            colored_mask[mask[inst] > 0] = color

        # Plot radar
        ax = axes[row, 0]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Radar idx={idx}")
        ax.axis("off")

        # Plot colored mask
        ax = axes[row, 1]
        ax.imshow(colored_mask)
        ax.set_title(f"Mask  idx={idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"Visualizing indices {INDICES} from {TENSORSTORE_PATH}")
    visualize_colored_masks(TENSORSTORE_PATH, INDICES)
