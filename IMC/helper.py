import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import os

def plot_batch_per_sample(batch, figsize=(15, 10), title = None, id = 1):
    """
    Plots each sample (set of images) in a row.

    Args:
        batch (Tensor): Tensor of shape (B, N, 1, H, W)
        figsize (tuple): Size of the figure.
        title (str): Optional title.
    """
    if not os.path.isdir("./tmp"):
        os.makedirs("./tmp")
    
    B, N, C, H, W = batch.shape
    fig, axes = plt.subplots(B, N, figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=16)

    for i in range(B):
        # Compute stats for sample i (all its N images)
        sample_pixels = batch[i].view(-1)
        stats_text = (
            f"min: {sample_pixels.min().item():.3f}\n"
            f"max: {sample_pixels.max().item():.3f}\n"
            f"mean: {sample_pixels.mean().item():.3f}\n"
            f"std: {sample_pixels.std().item():.3f}\n"
            f"median: {sample_pixels.median().item():.3f}"
        )

        # Plot images
        for j in range(N):
            img = batch[i, j].squeeze().cpu().numpy()
            ax = axes[i][j] if B > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            ax.axis('off')

        # Add stats text to left of the row (using the first image axis)
        # Adjust position to the left outside the image
        ax_stats = axes[i][0] if B > 1 else axes[0]
        ax_stats.text(
            -0.5, 0.5, stats_text,
            fontsize=10,
            va='center', ha='right',
            transform=ax_stats.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.savefig(f"./tmp/{id}.png", dpi=300, bbox_inches='tight')
    plt.close()


def normalize_per_sample(batch):
    """
    Normalize each sample in the batch independently:
    - Compute mean and std over all slices and pixels of that sample.
    - Normalize all slices in that sample with these stats.

    Args:
        batch (Tensor): shape (B, N, 1, H, W)

    Returns:
        Tensor: normalized batch, same shape as input
    """
    
    B, N, C, H, W = batch.shape
    # Compute mean and std per sample across all slices and pixels
    # Shape of mean/std: (B, 1, 1, 1, 1) to broadcast correctly
    mean = batch.view(B, -1).mean(dim=1).view(B, 1, 1, 1, 1)
    std = batch.view(B, -1).std(dim=1).view(B, 1, 1, 1, 1)
    
    # Avoid division by zero by clamping std to a minimum value (e.g. 1e-8)
    std = std.clamp(min=1e-8)
    
    # Normalize batch with broadcasting
    batch_norm = (batch - mean) / std

    return batch_norm
