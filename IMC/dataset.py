import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import torchvision.transforms as transforms
from collections import defaultdict


class MRIMultiSliceDataset(Dataset):
    def __init__(
        self,
        series_dirs,
        labels_dict,
        metadata_fields,
        num_slices=3,
        image_size=(224, 224),
        transform=None,
    ):
        """
        Args:
            series_dirs (list of str): paths to folders containing DICOM slices for each series
            labels_dict (dict): mapping series_dir -> dict with labels for sequence, plane, body, contrast
            metadata_fields (list of str): DICOM tags to extract as metadata features
            num_slices (int): number of slices to sample per series
            image_size (tuple): output image size (H, W)
            transform (torchvision transform): transform to apply to slices
        """
        self.series_dirs = series_dirs
        self.labels_dict = labels_dict
        self.metadata_fields = metadata_fields
        self.num_slices = num_slices
        self.image_size = image_size
        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5], [0.5]
                ),  # example normalization for 1 channel
            ]
        )

        # Metadata field processing info
        # Define which fields are categorical, which numeric (example)
        self.categorical_fields = [
            "SequenceName",
            "BodyPartExamined",
            "PatientPosition",
            "ScanningSequence",
        ]
        self.numeric_fields = ["EchoTime", "RepetitionTime", "FlipAngle"]

        # Build vocab dictionaries for categorical fields
        self.vocab_maps = self._build_vocab_maps()

    def _build_vocab_maps(self):
        # Scan all series to collect unique categorical values per field
        cat_values = defaultdict(set)
        for series_dir in self.series_dirs:
            # Use first DICOM in series for metadata extraction
            dcm_path = self._get_first_dicom_path(series_dir)
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            for field in self.categorical_fields:
                val = getattr(ds, field, None)
                if val is not None:
                    cat_values[field].add(val)

        vocab_maps = {}
        for field, values in cat_values.items():
            vocab_maps[field] = {v: i for i, v in enumerate(sorted(values))}
        return vocab_maps

    def _get_first_dicom_path(self, series_dir):
        files = [f for f in os.listdir(series_dir) if f.endswith(".dcm")]
        files.sort()
        return os.path.join(series_dir, files[0])

    def _load_slices(self, series_dir):
        # List all DICOM slice files
        files = [f for f in os.listdir(series_dir) if f.endswith(".dcm")]
        files.sort()
        n = len(files)
        if n == 0:
            raise RuntimeError(f"No DICOM files found in {series_dir}")

        # Sample num_slices indices evenly spaced or random
        if n <= self.num_slices:
            indices = list(range(n))  # take all if less slices
        else:
            step = n / self.num_slices
            indices = [int(i * step) for i in range(self.num_slices)]

        slices = []
        for idx in indices:
            path = os.path.join(series_dir, files[idx])
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)

            # Normalize pixel intensities to [0,1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            # Convert to 1-channel image (add channel dim)
            img = np.expand_dims(img, axis=0)  # (1, H, W)

            # Apply transforms (expects PIL image or tensor)
            img_tensor = self.transform(torch.from_numpy(img))
            slices.append(img_tensor)

        # Stack to tensor: (N_slices, C, H, W)
        slices_tensor = torch.stack(slices)
        return slices_tensor

    def _extract_metadata(self, series_dir):
        dcm_path = self._get_first_dicom_path(series_dir)
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

        meta_vector = []

        # Categorical encoding
        for field in self.categorical_fields:
            val = getattr(ds, field, None)
            if val is None:
                # Use unknown index -1 encoded as 0 vector or a special index (e.g. len(vocab))
                idx = len(self.vocab_maps.get(field, {}))
            else:
                idx = self.vocab_maps.get(field, {}).get(
                    val, len(self.vocab_maps.get(field, {}))
                )
            meta_vector.append(idx)

        # Numeric fields normalization - simple min-max normalization example (you can customize)
        # For simplicity, clip/normalize by predefined max values here (should be computed from dataset)
        max_vals = {"EchoTime": 500, "RepetitionTime": 5000, "FlipAngle": 180}
        for field in self.numeric_fields:
            val = getattr(ds, field, None)
            if val is None:
                norm_val = 0.0
            else:
                norm_val = float(val) / max_vals.get(field, 1.0)
                norm_val = np.clip(norm_val, 0, 1)
            meta_vector.append(norm_val)

        meta_vector = torch.tensor(meta_vector, dtype=torch.float32)
        return meta_vector

    def __len__(self):
        return len(self.series_dirs)

    def __getitem__(self, idx):
        series_dir = self.series_dirs[idx]

        # Load image slices
        image_slices = self._load_slices(series_dir)  # (N_slices, C, H, W)

        # Extract metadata vector
        metadata = self._extract_metadata(series_dir)  # (metadata_dim,)

        # Get labels from labels_dict
        label_dict = self.labels_dict[series_dir]
        sequence_label = label_dict["sequence"]  # int class index
        plane_label = label_dict["plane"]  # int class index
        body_label = label_dict["body"]  # int class index
        contrast_label = label_dict["contrast"]  # 0 or 1

        labels = {
            "sequence": torch.tensor(sequence_label, dtype=torch.long),
            "plane": torch.tensor(plane_label, dtype=torch.long),
            "body": torch.tensor(body_label, dtype=torch.long),
            "contrast": torch.tensor(contrast_label, dtype=torch.float32),
        }

        return image_slices, metadata, labels


# Example usage:

if __name__ == "__main__":
    # Dummy example of dataset construction

    # List of directories each containing DICOM series
    series_dirs = [
        "/path/to/series1",
        "/path/to/series2",
        "/path/to/series3",
    ]

    # Labels dictionary for each series
    labels_dict = {
        "/path/to/series1": {"sequence": 0, "plane": 1, "body": 2, "contrast": 1},
        "/path/to/series2": {"sequence": 3, "plane": 0, "body": 1, "contrast": 0},
        "/path/to/series3": {"sequence": 1, "plane": 2, "body": 0, "contrast": 1},
    }

    metadata_fields = [
        "SequenceName",
        "BodyPartExamined",
        "PatientPosition",
        "ScanningSequence",
        "EchoTime",
        "RepetitionTime",
        "FlipAngle",
    ]

    dataset = MRIMultiSliceDataset(
        series_dirs, labels_dict, metadata_fields, num_slices=3
    )

    # Test __getitem__
    image_slices, metadata, labels = dataset[0]

    print("Slices shape:", image_slices.shape)  # (N_slices, C, H, W)
    print("Metadata vector shape:", metadata.shape)
    print("Labels:", labels)
