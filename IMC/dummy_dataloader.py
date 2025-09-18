import torch
from torch.utils.data import Dataset, DataLoader

class DummyMRIDataset(Dataset):
    def __init__(self, num_samples=100, n_slices=3, img_channels=1, img_size=224,
                 metadata_dim=512,
                 num_classes_dict={"sequence": 5, "plane": 3, "body": 5, "contrast": 1}):
        self.num_samples = num_samples
        self.n_slices = n_slices
        self.img_channels = img_channels
        self.img_size = img_size
        self.metadata_dim = metadata_dim
        self.num_classes_dict = num_classes_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create dummy MRI slices: shape (N_slices, C, H, W)
        images = torch.randn(self.n_slices, self.img_channels, self.img_size, self.img_size)

        # Dummy metadata vector
        metadata = torch.randn(self.metadata_dim)

        # Dummy targets:
        # For multi-class classification targets, random integers in [0, num_classes-1]
        seq_t = torch.randint(0, self.num_classes_dict["sequence"], (1,)).squeeze()
        plane_t = torch.randint(0, self.num_classes_dict["plane"], (1,)).squeeze()
        body_t = torch.randint(0, self.num_classes_dict["body"], (1,)).squeeze()

        # For contrast (binary), random 0 or 1 tensor with shape ()
        contrast_t = torch.randint(0, 2, (1,)).float().squeeze()

        targets = (seq_t, plane_t, body_t, contrast_t)

        return images, metadata, targets


# Usage example:
if __name__ == "__main__":
    dummy_dataset = DummyMRIDataset(num_samples=200,  n_slices = 5, metadata_dim = 3*256)
    dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

    for batch_idx, (images, metadata, targets) in enumerate(dummy_loader):
        print(f"Batch {batch_idx}:")
        print(f"  images.shape = {images.shape}")          # (B, N_slices, C, H, W)
        print(f"  metadata.shape = {metadata.shape}")      # (B, metadata_dim)
        print(f"  targets shapes = {[t.shape for t in targets]}")
        
        print(targets)
        if batch_idx == 1:  # just show first 2 batches
            break
