import logging
from ast import literal_eval
from typing import Any, Dict, List, Tuple

import copy 
import random

import numpy as np
import pandas as pd
import torch
from google.cloud import storage  # type: ignore
from natsort import natsorted
from pydicom import FileDataset, dcmread
from pydicom.filebase import DicomBytesIO
from torch.utils.data import DataLoader, Dataset

from augment import augment
from dicom_tag_encoding import encode_dicom_tags_by_version

log = logging.getLogger("dataloader")


class LiverDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 100,
        n_slices: int = 3,
        img_channels: int = 1,
        img_size: int = 224,
        label_names: Dict[str, List[str]] = {
            "label_SequenceType": [
                "T1",
                "T2",
                "DWI",
                "ADC",
                "SUB",
                "DIXON_F",
                "DIXON_IN",
                "DIXON_OPP",
                "BOLUS",
                "OTHER",
                "na",
            ],
            "label_FatSat": ["yes", "no", "na"],
            "label_MRCP": ["yes", "no", "na"],
            "label_AcquisitionPlane": ["AX", "COR", "SAG", "ORTHO", "ROT", "na"],
            "label_ContrastPhase": ["pre", "art", "portven", "trans", "hepa", "na"],
            "label_Contrast": ["pre", "post", "na"],
            "label_Localizer": ["yes", "no", "na"],
        },
        label_path: str = "",
        augment_conf: str = "NONE2D",
    ):
        self.num_samples = num_samples
        self.n_slices = n_slices
        self.img_channels = img_channels
        self.img_size = img_size
        self.label_names = label_names
        self.augment_conf = augment_conf

        # gcs client
        self.gcs_client = storage.Client()
        
        # list holding the labels
        self.labels = []

        # List of dicom series
        self.path_list = []

        # Buffer for slice filenames
        self.slice_filenames={}

        # Buffer for images
        self.img_buffer = {}

        # populate label and image file path lists
        labels_df = pd.read_csv(label_path)
        labels_df = labels_df.set_index("Filepath")

        for index, row in labels_df.iterrows():
            # add file path to path list
            self.path_list.append(index)
            # get label idx:
            self.labels.append(row.to_dict())

        print("Available images: ", len(self.path_list))
        

    def __len__(self: Any) -> int:
        return self.num_samples

    def get_n_labels(self):
        r = {}
        for x in self.label_names:
            n = len(self.label_names[x])
            r[x] = n
        return r

    def get_cache_size(self):
        return len(self.img_buffer)
        
    def get_bucket_filelist(self, path_dicom_folder):
        bucket_name, folder_filename = path_dicom_folder[5:].split("/", 1)
        
        if folder_filename is not self.slice_filenames:       
            
            blobs = self.gcs_client.list_blobs(bucket_name, prefix=folder_filename)

            sfn = []
            for blob in blobs:
                sfn.append(blob.name)

            self.slice_filenames[folder_filename] = natsorted(sfn)
            
        return bucket_name, folder_filename, self.slice_filenames[folder_filename] 

                
    def open_dicom_slice_from_series(self, path_dicom_folder: str, sampling_type: str = "equidistant", stop_before_pixels: bool = True, n_images : int = 1) -> FileDataset:
        """Open a single slice given the path to a dicom folder (containing a whole
        dicom series).
    
        Parameters:
        -----------
        path_dicom_folder: str
            path to a folder containing a whole dicom series
            e.g. gs://bucket/folder/, http://url/folder or /local/folder/
        type:
            "random": select a random slice from the series
            "center": select the center slice from the series
            "first": select the first slice from the series
        stop_before_pixels: bool
            if True, the pixel data is not read, only the header
    
        Returns:
        --------
        slice_image: FileDataset
            the dicom slice as FileDataset, including the header, and optionally the pixel data
        """
        slice_filenames: List[str] = []

        if not  path_dicom_folder.startswith("gs://"):
            raise Exception("invalid GCP bucket provided")
            
        bucket_name, folder_filename, slice_filenames = self.get_bucket_filelist(path_dicom_folder)

        if not slice_filenames:
            raise RuntimeError(f"No DICOM files found in bucket path: {path_dicom_folder}")

        # --- get indices of slices that need to be loaded ---
        N = len(slice_filenames)
        slice_inds = []
        
        # offset computation
        f = N // n_images
        off = min(f//4,2)*n_images
        
        # sample distinct slices
        if n_images <= N:
            if sampling_type == "random":
                slice_inds = random.sample(range(off,N-off), n_images)
            else:
                slice_inds = [int(x) for x in np.linspace(off, N-1-off, n_images)]
        else:
            slice_inds = list(range(N)) + [None]*(n_images-N)

        # --- load slices from buffer or from the GCP bucket ---
        slice_images = []
        
        for slice_idx in slice_inds:
            # skip non-values. will be converted to black slices later
            if slice_idx is None:
                slice_images.append(None)
                continue
                
            # generate key for buffering
            fid = f"{bucket_name}-{slice_filenames[slice_idx]}"
            
            if not (fid in self.img_buffer):
                # read single slice:
                bucket = self.gcs_client.get_bucket(bucket_name)
                blob = bucket.blob(slice_filenames[slice_idx])
                file_obj = DicomBytesIO()
                blob.download_to_file(file_obj)
                file_obj.seek(0)

                # read dicom image
                dcm_image = dcmread(file_obj, stop_before_pixels=stop_before_pixels)

                # covert to numpy array and store in buffer
                img_arr = dcm_image.pixel_array.astype(np.float32)
                
                if not len(img_arr.shape) == 2:
                    raise Exception(f"ERROR: {fid} has invalid image dimensions ({len(img_arr.shape)})")
                    
                self.img_buffer[fid] = img_arr
       
            slice_images.append(copy.deepcopy(self.img_buffer[fid]))
        
        return slice_images
        

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        # --- IMAGES ---
        dcm_images = self.open_dicom_slice_from_series(self.path_list[idx], sampling_type="equidistant", stop_before_pixels=False, n_images = self.n_slices)
        image_list = []
        
        for image in dcm_images:
            
            # we dont accept images with channel dimension
            if (image is None) or (len(image.shape)>2):
                 image = np.zeros((244, 244), dtype=np.float32)
            
            # do augmentation
            image = augment(image, self.augment_conf)

            # conver to float32 torch tensor of shape (1, H, W)
            image_list.append(torch.Tensor(image).unsqueeze(0).to(torch.float32))
            
        images = torch.stack(image_list, dim=0)  # (N_slices, 1, H, W)

        # --- METADATA ---
        
        # creast single row dataframe containing all labels / meta related to indexed sample
        dicom_tags_df = pd.DataFrame([self.labels[idx]])

        # encode metadata:
        enc_meta_data_df = encode_dicom_tags_by_version(
            dicom_tags_df=dicom_tags_df,
            dicom_encoding_version="brain",
        )

        if len(enc_meta_data_df) == 0:
            raise Exception("ERROR: metadata encoding failed")
        
        # convert into torch Tensor
        metadata = torch.tensor(enc_meta_data_df.iloc[0].to_numpy(), dtype=torch.float32)
        

        # --- TARGETS ---
        label_idx_dict: Dict[str, int] = {}

        for label_class in self.label_names:
            if not label_class in self.labels[idx]:
                raise Exception(f"Missing Label! {label_class}")
                
            label_value = self.labels[idx][label_class]

            if label_value in self.label_names[label_class]:
                label_idx_dict[label_class] = self.label_names[label_class].index(label_value)
            else:
                label_idx_dict[label_class] = -1
                print(f"WARNING: Label value {label_value} not in label names of label class {label_class}, found {self.label_names[label_class]} only")

        targets = tuple(torch.tensor(label_idx_dict.get(label_class, -1)) for label_class in self.label_names.keys())
           
        return images, metadata, targets


# Usage example:
if __name__ == "__main__":
    dummy_dataset = LiverDataset(num_samples=200, n_slices=5, label_path="~/pvai_labels_20250603.csv")
    dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

    n_label = dummy_dataset.get_n_labels()
    print(n_label)
    
    for batch_idx, (images, metadata, targets) in enumerate(dummy_loader):
        print(f"Batch {batch_idx}:")
        print(f"  images.shape = {images.shape}")  # (B, N_slices, C, H, W)
        print(f"  metadata.shape = {metadata.shape}")  # (B, metadata_dim)
        print(f"  targets shapes = {[t.shape for t in targets]}")

        print(metadata)
        print(targets)
        if batch_idx == 4:  # just show first 2 batches
            break
