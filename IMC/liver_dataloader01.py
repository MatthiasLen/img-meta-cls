import glob
import logging
from ast import literal_eval
from typing import Any, Dict, List, Tuple


import copy 
import random

import elasticdeform
import numpy as np
import pandas as pd
import torch
from google.cloud import storage  # type: ignore
from natsort import natsorted
from pydicom import FileDataset, dcmread
from pydicom.filebase import DicomBytesIO
from skimage.filters import gaussian
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("augment")


# predefined augmentation configurations:

DEFAULT2D = {
    "patch_size": 224,
    "crop": "random_center",
    "flip": True,
    "rot90": True,
    "elastic": {"num_control_points": 7, "sigma_frac": 0.1, "rate": 0.4},
    "noise": {"max_std": 0.1, "rate": 0.4, "fixed": False},
    "gamma": {"min_log_gamma": -0.2231, "max_log_gamma": 0.1823, "rate": 0.4},
    "blur": {"max_sigma": 2.0, "rate": 0.4, "fixed": False},
    "project": "none",
    "normalize": "none",
}

ZSCOREDEFAULT2D = {
    "patch_size": 224,
    "crop": "random_center",
    "flip": True,
    "rot90": True,
    "elastic": {"num_control_points": 7, "sigma_frac": 0.1, "rate": 0.4},
    "noise": {"max_std": 0.1, "rate": 0.4, "fixed": False},
    "gamma": {"min_log_gamma": -0.2231, "max_log_gamma": 0.1823, "rate": 0.4},
    "blur": {"max_sigma": 2.0, "rate": 0.4, "fixed": False},
    "project": "none",
    "normalize": "zscore",
}


NONE2D = {
    "patch_size": 224,
    "crop": "center",
    "flip": False,
    "rot90": False,
    "elastic": {"num_control_points": 7, "sigma_frac": 0.1, "rate": 0.0},
    "noise": {"max_std": 0.1, "rate": 0.0, "fixed": False},
    "gamma": {"min_log_gamma": 0.0, "max_log_gamma": 0.0, "rate": 0.0},
    "blur": {"max_sigma": 2.0, "rate": 0.0, "fixed": False},
    "project": "none",
    "normalize": "none",
}

ZSCORENONE2D = {
    "patch_size": 224,
    "crop": "center",
    "flip": False,
    "rot90": False,
    "elastic": {"num_control_points": 7, "sigma_frac": 0.1, "rate": 0.0},
    "noise": {"max_std": 0.1, "rate": 0.0, "fixed": False},
    "gamma": {"min_log_gamma": 0.0, "max_log_gamma": 0.0, "rate": 0.0},
    "blur": {"max_sigma": 2.0, "rate": 0.0, "fixed": False},
    "project": "none",
    "normalize": "zscore",
}


def find_word(string_list: Any, list_word_to_find: List[str], flag=False):  # type: ignore[no-untyped-def]
    """Checks if an exact word is present in any of the strings in the given
    list.

    Args:
        string_list (Any): List of strings or single string to search for the word.
        list_word_to_find (str): List of words to search for in the list of strings.
        flag (bool): A flag that toggles its value if the token is found. Defaults to True.
    Returns:
        bool: The toggled value of the flag if the exact word is found, otherwise the original flag value.
    """
    if string_list is None or not isinstance(string_list, (list, str)):
        return flag

    if isinstance(string_list, str):
        string_list = [string_list]

    for word in string_list:
        for word_to_find in list_word_to_find:
            if word == word_to_find:
                flag = not flag
                break

    return flag


def crop(image: np.ndarray, patch_size: int, crop: str, dims: int = 2) -> np.ndarray:
    """Crop a patch of size patch_size x patch_size (x patch_size) from the
    image. If the image is smaller than patch_size in any dimension, it is
    padded with zeros first.

    Parameters:
    -----------
    image: np.ndarray
        The input image to be cropped. Can be 2D or 3D.
    patch_size: int
        The size of the patch to be cropped.
    crop: str
        The type of crop to be performed. Can be "random", "center", or "random_center".
    dims: int
        The number of dimensions to crop to. Can be 2 or 3.
    """
    s = image.shape

    # pad first
    if s[0] < patch_size:
        p01 = np.maximum(0, (patch_size - s[0]) // 2)
        p02 = np.maximum(0, (patch_size - s[0]) // 2 + (patch_size - s[0]) % 2)
    else:
        p01 = 0
        p02 = 0
    if s[1] < patch_size:
        p11 = np.maximum(0, (patch_size - s[1]) // 2)
        p12 = np.maximum(0, (patch_size - s[1]) // 2 + (patch_size - s[1]) % 2)
    else:
        p11 = 0
        p12 = 0
    if len(s) == 3 and dims == 3 and s[2] < patch_size:
        p21 = np.maximum(0, (patch_size - s[2]) // 2)
        p22 = np.maximum(0, (patch_size - s[2]) // 2 + (patch_size - s[2]) % 2)
    else:
        p21 = 0
        p22 = 0

    if len(s) == 2 and (s[0] < patch_size or s[1] < patch_size):
        image_padded = np.pad(image, ((p01, p02), (p11, p12)), mode="constant", constant_values=0)
    elif len(s) == 3 and (s[0] < patch_size or s[1] < patch_size or s[2] < patch_size):
        image_padded = np.pad(image, ((p01, p02), (p11, p12), (p21, p22)), mode="constant", constant_values=0)
    else:
        image_padded = image

    s = image_padded.shape

    # now crop:
    if dims == 2:
        if crop == "random_center":
            x = np.random.randint(s[0] - patch_size + 1)
            y = np.random.randint(s[1] - patch_size + 1)
            if len(s) == 3:
                # sample with normal distribution around center slice ()
                z = int(np.round(np.random.normal(loc=s[2] // 2, scale=0.1 * s[2])).clip(0, s[2]))
        elif crop == "center":  # center crop
            x = s[0] // 2 - patch_size // 2
            y = s[1] // 2 - patch_size // 2
            if len(s) == 3:
                z = s[2] // 2
        else:
            raise Exception(f"Unknown crop parameter {crop}")

        if len(s) == 2:
            cropped_image = image_padded[x : x + patch_size, y : y + patch_size]
            return cropped_image
        elif len(s) == 3:
            log.debug(f"selected slice: {z}/{s[2]}")
            cropped_image = image_padded[x : x + patch_size, y : y + patch_size, z]
            return cropped_image
        else:
            raise Exception(f"Image has {len(s)} dimensions!")

    elif dims == 3:
        if crop == "random":
            x = np.random.randint(s[0] - patch_size + 1)
            y = np.random.randint(s[1] - patch_size + 1)
            z = np.random.randint(s[2] - patch_size + 1)
        elif crop == "center":  # center crop
            x = s[0] // 2 - patch_size // 2
            y = s[1] // 2 - patch_size // 2
            z = s[2] // 2 - patch_size // 2
        elif crop == "random_center":
            x = np.random.randint(s[0] - patch_size + 1)
            y = np.random.randint(s[1] - patch_size + 1)
            # sample with normal distribution around center slice
            z = int(np.round(np.random.normal(loc=s[2] // 2, scale=0.1 * s[2])).clip(0, s[2] - patch_size + 1))
        else:
            raise Exception(f"Unknown crop parameter {crop} for {dims}D crop")

        # return cropped volume patch
        cropped_image = image_padded[x : x + patch_size, y : y + patch_size, z : z + patch_size]
        return cropped_image

    else:
        raise Exception(
            f"Image has {len(s)} dimensions, dims is {dims}: only dims=2 or dims=3 are supported for cropping!"
        )


def project(image: np.ndarray, method: str = "none") -> np.ndarray:
    """Project a 3D image to 2D using the specified method.

    Parameters:
    -----------
    image: np.ndarray
        The input 3D image to be projected.
    method: str
        The projection method to be used. Can be one of the following:
        "x_y_z_maxx_maxy_maxz":
            project to x-y, y-z, and x-z planes at center slice, and
             maximum intensity projections along x, y, and z axes
        "none":
            no projection, return the image as is
    """
    if method == "x_y_z_maxx_maxy_maxz":
        # project to x-y plane at center slice
        assert len(image.shape) == 3, "Image must be 3D for x_y_z_maxx_maxy_maxz projection"
        cx = image.shape[0] // 2 + 1
        cy = image.shape[1] // 2 + 1
        cz = image.shape[2] // 2 + 1

        slice_x = image[cx, :, :]
        slice_y = image[:, cy, :]
        slice_z = image[:, :, cz]

        mip_x = np.max(image, axis=0)
        mip_y = np.max(image, axis=1)
        mip_z = np.max(image, axis=2)

        projected_image = np.stack((slice_x, slice_y, slice_z, mip_x, mip_y, mip_z), axis=0)
        log.debug("using orthogonal slices and maximum intensity projections")
        return projected_image
    elif method == "none":
        return image
    else:
        raise ValueError(f"Unknown projection method: {method}")


def rot90(image: np.ndarray) -> np.ndarray:
    """Rotate the image by a random multiple of 90 degrees."""
    nr_rotations = np.random.randint(4)
    image = np.rot90(image, k=nr_rotations, axes=(0, 1))
    log.debug(f"rotated image {nr_rotations} times 90 degrees")
    return image


def flip(image: np.ndarray) -> np.ndarray:
    """Flip the image randomly along each axis."""
    if np.random.rand() < 0.5:
        # flip image horizontally
        image = np.flip(image, axis=1)
        log.debug("flipped image horizontally")
    if np.random.rand() < 0.5:
        # flip image vertically
        image = np.flip(image, axis=0)
        log.debug("flipped image vertically")
    if len(image.shape) == 3 and np.random.rand() < 0.5:
        # flip image in z-direction
        image = np.flip(image, axis=2)
        log.debug("flipped image in z-direction")
    return image


def blur(image: np.ndarray, max_sigma: float = 2.0, rate: float = 0.4, fixed: bool = False) -> np.ndarray:
    """Apply Gaussian blur to the image with a random sigma value."""

    # don't blur each time
    if np.random.rand() < rate:
        if fixed:
            sigma = 1.5
        else:
            sigma = 1.0 + np.random.rand() * (max_sigma - 1.0)

        # sample sigma between 1.0 and max_sigma - 1.0
        image = gaussian(image, sigma=sigma)
        log.debug(f"blurred image with sigma {sigma}")
    return image


def noise(image: np.ndarray, max_std: float = 0.1, rate: float = 0.4, fixed: bool = False) -> np.ndarray:
    """Add Gaussian noise to the image with a random standard deviation."""

    # don't add noise each time:
    if np.random.rand() < rate:
        # generate noise with mean = 0.0
        if fixed:
            np.random.seed(1234)
        noise = np.random.randn(*image.shape)

        # scale by random value below max_std
        scale = 1.0 if fixed else np.random.rand()
        std = (image.max() - image.min()) * max_std * scale
        # add noise
        image = image + (noise * std)
        log.debug(f"added noise with std {std}")
    return image


def elastic(image: np.ndarray, num_control_points: int = 7, sigma_frac: float = 0.1, rate: float = 0.4) -> np.ndarray:
    """Apply elastic deformation to the image."""
    if np.random.rand() < rate:
        # sigma is in pixels
        max_shape = np.max(np.array(image.shape))
        sigma = sigma_frac * max_shape / num_control_points

        # apply deformation with a random 3 x 3 (x 3 ) grid
        # elasticdeform uses spline interpolation
        # default is order 3, but order 1 guarantees values stay inside previous intensity range
        deformed_image = elasticdeform.deform_random_grid(image, sigma=sigma, points=num_control_points, order=1)
        log.debug(f"applied elastic deformation with sigma {sigma} and {num_control_points} control points")
        return deformed_image
    else:
        return image


def gamma(
    image: np.ndarray, min_log_gamma: float = -0.2231, max_log_gamma: float = 0.1823, rate: float = 0.4
) -> np.ndarray:
    """Apply gamma transformation to the image with a random gamma value."""

    # don't always apply gamma:
    if np.random.rand() < rate:
        # normalize to range 0.0 to 1.0, because this is how gamma transforms can be safely applied
        # and stay within original range
        image_min = image.min()
        image_max = image.max()
        norm_image = image - image_min
        if image_max > image_min:
            norm_image /= image_max - image_min

        log_gamma = min_log_gamma + (max_log_gamma - min_log_gamma) * np.random.rand()
        gamma = np.exp(log_gamma)
        distorted_image = np.power(norm_image, gamma)
        log.debug(f"applied gamma transformation with gamma {gamma}")

        # normalize back to original intensity range:
        if image_max > image_min:
            renorm_image = distorted_image * (image_max - image_min)
        else:
            renorm_image = distorted_image
        renorm_image += image_min

        return renorm_image
    else:
        return image


# main augmentation routine
def augment(image: np.ndarray, augment_conf: str = "DEFAULT2D") -> np.ndarray:
    """Augment the image using the specified augmentation configuration.
    Possible configurations are defined as constants at the beginning of this
    file.

    The configuration is a dictionary with the following keys:
    - patch_size: int, size of the patch to be cropped
    - crop: str, type of crop to be performed, can be "random", "
    "center", or "random_center"
    - flip: bool, whether to flip the image randomly along each axis
    - rot90: bool, whether to rotate the image by a random multiple of 90 degrees
    - elastic: dict, parameters for elastic deformation, with keys:
        - num_control_points: int, number of control points for the deformation grid
        - sigma_frac: float, fraction of the maximum image dimension to be used as sigma
        - rate: float, probability of applying the deformation
    - noise: dict, parameters for adding Gaussian noise, with keys:
        - max_std: float, maximum standard deviation of the noise as a fraction of the image
            intensity range (max - min)
        - rate: float, probability of adding noise
        - fixed: bool, whether to use a fixed random seed for noise generation
    - gamma: dict, parameters for gamma transformation, with keys:
        - min_log_gamma: float, minimum log gamma value
        - max_log_gamma: float, maximum log gamma value
        - rate: float, probability of applying the transformation
    - blur: dict, parameters for Gaussian blur, with keys:
        - max_sigma: float, maximum sigma value for the Gaussian kernel
        - rate: float, probability of applying the blur
        - fixed: bool, whether to use a fixed sigma value
    - project: str, projection method to be used, can be one of the following:
        - "x_y_z_maxx_maxy_maxz": project to x-y, y-z, and x-z planes at center slice,
            and maximum intensity projections along x, y, and z axes
        - "none": no projection, return the image as is
    - normalize: str, normalization method to be used, can be one of the following:
        - "zscore": normalize to z-score (mean 0, std 1)
        - "none": no normalization
    """

    log.debug(f"== augmenting image with config {augment_conf} ==")

    augment_dict = globals().get(augment_conf)

    dims = 2
    if augment_dict["project"] != "none":
        # if projection is used, we need to crop a 3D patch
        dims = 3
    image = crop(image, augment_dict["patch_size"], augment_dict["crop"], dims=dims)

    if augment_dict["flip"]:
        image = flip(image)

    if augment_dict["rot90"]:
        image = rot90(image)

    if augment_dict["normalize"] == "zscore":
        # normalize to z-score
        image_mean = image.mean()
        image_std = image.std()
        image = image - image_mean
        if image_std > 0:
            image = image / image_std

    image = blur(image, **augment_dict["blur"])

    image = noise(image, **augment_dict["noise"])

    image = elastic(image, **augment_dict["elastic"])

    image = gamma(image, **augment_dict["gamma"])

    image = project(image, augment_dict["project"])

    return image


def encode_angio_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the 'AngioFlag' column of the provided DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing series information.
            Must be a pandas DataFrame with an 'AngioFlag' column.
    Returns:
        pd.DataFrame: DataFrame with encoded AngioFlag.
    """
    # Create a new column for encoded AngioFlag

    if "AngioFlag" not in df.columns:
        df.insert(column="AngioFlag", value=np.nan, loc=len(df.columns))
    df["enc_AngioFlag"] = df["AngioFlag"].apply(lambda x: 1 if x == "Y" else np.nan)
    df["enc_AngioFlag"] = df["AngioFlag"].apply(lambda x: 0 if x == "N" else np.nan)

    return df.drop(columns=["AngioFlag"], errors="ignore")


def encode_series_description(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the 'SeriesDescription' column of the provided masterfile
    DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing series information.
            Must be a pandas DataFrame with a 'SeriesDescription' column.
    Returns:
        pd.DataFrame: DataFrame with encoded series descriptions.
    """
    if "SeriesDescription" not in df.columns:
        df.insert(column="SeriesDescription", value=np.nan, loc=len(df.columns))

    # Define the strings to search for in SeriesDescription
    loc_strings = ["loc", "scout", "survey"]
    mrcp_strings = ["mrcp"]
    bol_strings = ["bolus"]
    t1w_strings = ["mprage", "mp-rage", "t1"]
    asl_strings = ["asl"]
    flair_strings = ["flair"]
    b0b1_strings = ["field"]

    # Convert SeriesDescription to lowercase for consistent matching
    df["SeriesDescription"] = df["SeriesDescription"].astype(str).str.lower()

    # Ensure we only process string values in SeriesDescription
    df["SeriesDescription"] = df["SeriesDescription"].apply(lambda x: x if isinstance(x, str) else "")

    # Check if localizer string is in series description tag
    df["enc_loc_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in loc_strings) else 0
    )

    # Check if mrcp string is in series description tag
    df["enc_mrcp_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in mrcp_strings) else 0
    )

    # Check if bolus string is in series description tag
    df["enc_bolus_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in bol_strings) else 0
    )

    # Check if t1w string is in series description tag
    df["enc_t1w_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in t1w_strings) else 0
    )

    # Check if arterial spin labelling string is in series description tag
    df["enc_asl_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in asl_strings) else 0
    )

    # Check if flair string is in series description tag
    df["enc_flair_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in flair_strings) else 0
    )

    # Check if arterial spin labelling string is in series description tag
    df["enc_asl_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in asl_strings) else 0
    )

    # Check if any field mapping string is in series description tag
    df["enc_b0b1_SeriesDescription"] = df["SeriesDescription"].apply(
        lambda x: 1 if any(loc in x for loc in b0b1_strings) else 0
    )

    # set value to 2, if SeriesDescrption was nan
    df.loc[df["SeriesDescription"] == "NaN", "enc_bolus_SeriesDescription"] = 2
    df.loc[df["SeriesDescription"] == "NaN", "enc_loc_SeriesDescription"] = 2
    df.loc[df["SeriesDescription"] == "NaN", "enc_mrcp_SeriesDescription"] = 2

    return df.drop(columns=["SeriesDescription"])


def encode_image_type(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the 'ImageType' column of the provided masterfile DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing series information.
            Must be a pandas DataFrame with a 'ImageType' column.
    Returns:
        pd.DataFrame: DataFrame with encoded ImageType.
    """

    # Define the strings to search for in ImageType
    projection_string = ["PROJECTION", "MIP"]
    se_strings = ["SE", "M_SE"]
    epi_strings = ["EP", "EPI"]
    derived_strings = ["DERIVED", "TRACE"]
    diffusion_strings = ["DIFFUSION"]
    adc_strings = ["ADC"]
    dixon_water_strings = ["WATER", "W"]
    dixon_fat_strings = ["FAT", "F"]
    dixon_in_phase_strings = ["IN_PHASE", "IP"]
    dixon_opp_phase_strings = ["OPP_PHASE", "OUT_PHASE", "OP"]
    asl_strings = ["ASL"]
    ir_strings = ["IR"]
    field_echo_strings = ["FFE", "M_FFE"]
    subtraction_strings = ["SUB"]

    if "ImageType" not in df.columns:
        df.insert(column="ImageType", value=np.nan, loc=len(df.columns))

    # Check if projection string is in ImageType tag
    df["enc_projection_ImageType"] = df["ImageType"].apply(
        lambda x: 1 if isinstance(x, str) and any(word in x for word in projection_string) else 0
    )

    # Check if se string is in ImageType tag
    df["enc_se_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=se_strings)
    df["enc_se_ImageType"] = df["enc_se_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if epi string is in ImageType tag
    df["enc_epi_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=epi_strings)
    df["enc_epi_ImageType"] = df["enc_epi_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if derived string is in ImageType tag
    df["enc_derived_ImageType"] = df["ImageType"].apply(
        lambda x: 1 if isinstance(x, str) and any(word in x for word in derived_strings) else 0
    )

    # Check if diffusion string is in ImageType tag
    df["enc_diff_ImageType"] = df["ImageType"].apply(
        lambda x: 1 if isinstance(x, str) and any(word in x for word in diffusion_strings) else 0
    )

    # Check if adc string is in ImageType tag
    df["enc_adc_ImageType"] = df["ImageType"].apply(
        lambda x: 1 if isinstance(x, str) and any(word in x for word in adc_strings) else 0
    )

    # Check if dixon water string is in ImageType tag
    df["enc_dixw_ImageType"] = df["ImageType"].apply(
        lambda x: 1 if isinstance(x, str) and any(word in x for word in dixon_water_strings) else 0
    )

    # Check if dixon fat string is in ImageType tag
    df["enc_dixf_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=dixon_fat_strings)
    df["enc_dixf_ImageType"] = df["enc_dixf_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if dixon in phase string is in ImageType tag
    df["enc_dixip_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=dixon_in_phase_strings)
    df["enc_dixip_ImageType"] = df["enc_dixip_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if dixon opp phase string is in ImageType tag
    df["enc_dixop_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=dixon_opp_phase_strings)
    df["enc_dixop_ImageType"] = df["enc_dixop_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if arterial spin labelling string is in ImageType tag
    df["enc_asl_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=asl_strings)
    df["enc_asl_ImageType"] = df["enc_asl_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if inversion recovery string is in ImageType tag
    df["enc_ir_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=ir_strings)
    df["enc_ir_ImageType"] = df["enc_ir_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if field echo string is in ImageType tag
    df["enc_fe_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=field_echo_strings)
    df["enc_fe_ImageType"] = df["enc_fe_ImageType"].apply(lambda x: 1 if x else 0)

    # Check if subtraction string is in ImageType tag
    df["enc_sub_ImageType"] = df["ImageType"].apply(find_word, list_word_to_find=subtraction_strings)
    df["enc_sub_ImageType"] = df["enc_sub_ImageType"].apply(lambda x: 1 if x else 0)

    # overwrite with 2, if input was nan
    df.loc[df["ImageType"] == "NaN", "enc_projection_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_se_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_epi_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_derived_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_diff_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_adc_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_dixw_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_dixf_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_dixip_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_dixop_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_asl_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_ir_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_fe_ImageType"] = 2
    df.loc[df["ImageType"] == "NaN", "enc_sub_ImageType"] = 2

    return df.drop(columns=["ImageType"])


def encode_scan_option(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the 'ScanOption' column of the provided masterfile DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing series information.
            Must be a pandas DataFrame with a 'ScanOption' column.
    Returns:
        pd.DataFrame: DataFrame with encoded ScanOption.
    """

    if "ScanOptions" not in df.columns:
        df.insert(column="ScanOptions", value=np.nan, loc=len(df.columns))

    # Define the strings to search for in ScanOption
    fat_sat = ["FS", "SFS"]

    # Check if dixon opp phase string is in ImageType tag
    df["enc_FatSat_ScanOptions"] = df["ScanOptions"].apply(find_word, list_word_to_find=fat_sat)
    df["enc_FatSat_ScanOptions"] = df["enc_FatSat_ScanOptions"].apply(lambda x: 1 if x else 0)

    # overwrite with 2, if input was nan
    df.loc[df["ScanOptions"] == "NaN", "enc_FatSat_ScanOptions"] = 2

    return df.drop(columns=["ScanOptions"])


def encode_scanning_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the 'ScanningSequence' column of the provided masterfile
    DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing series information.
            Must be a pandas DataFrame with a 'ScanningSequence' column.
    Returns:
        pd.DataFrame: DataFrame with encoded ScanningSequence.
    """
    # Define the strings to search for in ScanningSequence
    se_string = ["SE"]
    ir_string = ["IR"]
    gr_string = ["GR"]
    ep_string = ["EP"]
    rm_string = ["RM"]

    if "ScanningSequence" not in df.columns:
        df.insert(column="ScanningSequence", value=np.nan, loc=len(df.columns))

    df["enc_se_ScanningSequence"] = df["ScanningSequence"].apply(find_word, list_word_to_find=se_string)
    df["enc_se_ScanningSequence"] = df["enc_se_ScanningSequence"].apply(lambda x: 1 if x else 0)

    df["enc_ir_ScanningSequence"] = df["ScanningSequence"].apply(find_word, list_word_to_find=ir_string)
    df["enc_ir_ScanningSequence"] = df["enc_ir_ScanningSequence"].apply(lambda x: 1 if x else 0)

    df["enc_gr_ScanningSequence"] = df["ScanningSequence"].apply(find_word, list_word_to_find=gr_string)
    df["enc_gr_ScanningSequence"] = df["enc_gr_ScanningSequence"].apply(lambda x: 1 if x else 0)

    df["enc_ep_ScanningSequence"] = df["ScanningSequence"].apply(find_word, list_word_to_find=ep_string)
    df["enc_ep_ScanningSequence"] = df["enc_ep_ScanningSequence"].apply(lambda x: 1 if x else 0)

    df["enc_rm_ScanningSequence"] = df["ScanningSequence"].apply(find_word, list_word_to_find=rm_string)
    df["enc_rm_ScanningSequence"] = df["enc_rm_ScanningSequence"].apply(lambda x: 1 if x else 0)

    # overwrite with 2, if input was nan
    df.loc[df["ScanningSequence"] == "NaN", "enc_se_ScanningSequence"] = 2
    df.loc[df["ScanningSequence"] == "NaN", "enc_ir_ScanningSequence"] = 2
    df.loc[df["ScanningSequence"] == "NaN", "enc_gr_ScanningSequence"] = 2
    df.loc[df["ScanningSequence"] == "NaN", "enc_ep_ScanningSequence"] = 2
    df.loc[df["ScanningSequence"] == "NaN", "enc_rm_ScanningSequence"] = 2

    return df.drop(columns=["ScanningSequence"])


def encode_sequence_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the 'SequenceVariant' column of the provided masterfile
    DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing series information.
            Must be a pandas DataFrame with a 'SequenceVariant' column.
    Returns:
        pd.DataFrame: DataFrame with encoded SequenceVariant.
    """

    # Define the strings to search for in SequenceVariant
    sk_string = ["SK"]
    mtc_string = ["MTC"]
    ss_string = ["SS"]
    trss_string = ["TRSS"]
    sp_string = ["SP"]
    mp_string = ["MP"]
    osp_string = ["OSP"]

    if "SequenceVariant" not in df.columns:
        df.insert(column="SequenceVariant", value=np.nan, loc=len(df.columns))

    df["enc_sk_SequenceVariant"] = df["SequenceVariant"].apply(find_word, list_word_to_find=sk_string)
    df["enc_sk_SequenceVariant"] = df["enc_sk_SequenceVariant"].apply(lambda x: 1 if x else 0)

    df["enc_mtc_SequenceVariant"] = df["SequenceVariant"].apply(find_word, list_word_to_find=mtc_string)
    df["enc_mtc_SequenceVariant"] = df["enc_mtc_SequenceVariant"].apply(lambda x: 1 if x else 0)

    df["enc_ss_SequenceVariant"] = df["SequenceVariant"].apply(find_word, list_word_to_find=ss_string)
    df["enc_ss_SequenceVariant"] = df["enc_ss_SequenceVariant"].apply(lambda x: 1 if x else 0)

    df["enc_trss_SequenceVariant"] = df["SequenceVariant"].apply(find_word, list_word_to_find=trss_string)
    df["enc_trss_SequenceVariant"] = df["enc_trss_SequenceVariant"].apply(lambda x: 1 if x else 0)

    df["enc_sp_SequenceVariant"] = df["SequenceVariant"].apply(find_word, list_word_to_find=sp_string)
    df["enc_sp_SequenceVariant"] = df["enc_sp_SequenceVariant"].apply(lambda x: 1 if x else 0)

    df["enc_mp_SequenceVariant"] = df["SequenceVariant"].apply(find_word, list_word_to_find=mp_string)
    df["enc_mp_SequenceVariant"] = df["enc_mp_SequenceVariant"].apply(lambda x: 1 if x else 0)

    df["enc_osp_SequenceVariant"] = df["SequenceVariant"].apply(find_word, list_word_to_find=osp_string)
    df["enc_osp_SequenceVariant"] = df["enc_osp_SequenceVariant"].apply(lambda x: 1 if x else 0)

    # overwrite with 2, if input was nan
    df.loc[df["SequenceVariant"] == "NaN", "enc_sk_SequenceVariant"] = 2
    df.loc[df["SequenceVariant"] == "NaN", "enc_mtc_SequenceVariant"] = 2
    df.loc[df["SequenceVariant"] == "NaN", "enc_ss_SequenceVariant"] = 2
    df.loc[df["SequenceVariant"] == "NaN", "enc_trss_SequenceVariant"] = 2
    df.loc[df["SequenceVariant"] == "NaN", "enc_sp_SequenceVariant"] = 2
    df.loc[df["SequenceVariant"] == "NaN", "enc_mp_SequenceVariant"] = 2
    df.loc[df["SequenceVariant"] == "NaN", "enc_osp_SequenceVariant"] = 2

    return df.drop(columns=["SequenceVariant"])


def encode_mr_acquisition_type(df: pd.DataFrame) -> pd.DataFrame:
    if "MRAcquisitionType" not in df.columns:
        df.insert(column="MRAcquisitionType", value=np.nan, loc=len(df.columns))

    mr_acquisition_type = df["MRAcquisitionType"]

    # Create a new column for encoded ImageOrientationPatient
    df["enc_1D_MRAcquisitionType"] = mr_acquisition_type.apply(lambda x: 1 if x == "1D" else 0)
    df["enc_2D_MRAcquisitionType"] = mr_acquisition_type.apply(lambda x: 1 if x == "2D" else 0)
    df["enc_3D_MRAcquisitionType"] = mr_acquisition_type.apply(lambda x: 1 if x == "3D" else 0)

    # overwrite with 2, if input was nan
    df.loc[df["MRAcquisitionType"] == "NaN", "enc_1D_MRAcquisitionType"] = 2
    df.loc[df["MRAcquisitionType"] == "NaN", "enc_2D_MRAcquisitionType"] = 2
    df.loc[df["MRAcquisitionType"] == "NaN", "enc_3D_MRAcquisitionType"] = 2

    return df.drop(columns=["MRAcquisitionType"])


def compute_orientation(image_ori: Any) -> str:
    """Compute orientation of dicom image based on ImageOrientationPatient tag.

    Args:
        image_ori (Any): tuple containing values of ImageOrientationPatient tag

    Returns:
        str: string indicating acquisition plane SAG -> sagittal, COR->coronal, AX-> axial, NA if image_ori not available
    """

    if image_ori is not None:
        # vector components along y axis [yx, yy, yz]
        image_y = np.array([image_ori[0], image_ori[1], image_ori[2]])
        # vector components along x axis [xx, xy, xz]
        image_x = np.array([image_ori[3], image_ori[4], image_ori[5]])

        # compute projection along z axis
        image_z = np.cross(image_x, image_y)
        abs_image_z = abs(image_z)

        # find closest unit vector to the nomal
        main_index = list(abs_image_z).index(max(abs_image_z))

        if main_index == 0:
            plane = "SAG"
        elif main_index == 1:
            plane = "COR"
        elif main_index == 2:
            plane = "AX"
        else:
            plane = "NA"

    else:
        plane = "NA"

    return plane


def parse_tag_value(tag_value: Any) -> Any:
    # if not a list and cannot be parsed to float or int, return as is
    # e.g. for strings
    # parse string values:
    # print("parsing: ", tag_value)
    if isinstance(tag_value, str):
        if tag_value.startswith(" "):
            # ignore space at the beginning
            # print("parsing again without leading space: ", tag_value[1:])
            return parse_tag_value(tag_value[1:])
        elif tag_value.startswith("[") and tag_value.endswith("]"):
            list_str_with_commas = tag_value[1:-1]
            if list_str_with_commas.startswith(" "):
                # ignore space at the beginning
                # print("parsing again without leading space: [", list_str_with_commas[1:], "]")
                return parse_tag_value("[" + list_str_with_commas[1:] + "]")
                list_str_with_commas = list_str_with_commas[1:]
            if list_str_with_commas.startswith("["):
                # find end of list
                end_nested_list = list_str_with_commas.index("]")
                # if there is a comma, continue after the comma
                if list_str_with_commas[end_nested_list + 1] == ",":
                    # print("skipping comma: ", list_str_with_commas[end_nested_list + 1])
                    rest = list_str_with_commas[end_nested_list + 2 :]
                else:
                    # print("no comma found")
                    rest = list_str_with_commas[end_nested_list + 1 :]
                # print("parse nested list: ", list_str_with_commas[0: end_nested_list+1])
                # print("parse rest: ", rest)
                return [parse_tag_value(list_str_with_commas[0 : end_nested_list + 1])] + parse_tag_value(
                    "[" + rest + "]"
                )
            elif list_str_with_commas.startswith("'"):
                # find end of string
                end_nested_string = list_str_with_commas[1:].index("'") + 1
                # if the str continues with a comma, parse the rest after the comma
                if (
                    end_nested_string + 1 < len(list_str_with_commas)
                    and list_str_with_commas[end_nested_string + 1] == ","
                ):
                    rest = list_str_with_commas[end_nested_string + 2 :]
                    # print("parse nested string: ", list_str_with_commas[1: end_nested_string])
                    # print("found comma, parse rest: ", rest)
                    return [parse_tag_value(list_str_with_commas[1:end_nested_string])] + parse_tag_value(
                        "[" + rest + "]"
                    )
                else:  # otherwise return parsed nested string in a list
                    rest = list_str_with_commas[end_nested_string + 1 :]
                    if len(rest) == 0:
                        # print("parse nested string without rest: ", list_str_with_commas[1: end_nested_string])
                        return [parse_tag_value(list_str_with_commas[1:end_nested_string])]
                    else:
                        # print("parse nested string: ", list_str_with_commas[1: end_nested_string])
                        # print("parse rest: ", rest)
                        return [parse_tag_value(list_str_with_commas[1:end_nested_string])] + parse_tag_value(
                            "[" + rest + "]"
                        )
            elif ", " in list_str_with_commas:
                # find next comma
                next_comma = list_str_with_commas.index(",")
                rest = list_str_with_commas[next_comma + 1 :]
                # print("parsing next element: ", list_str_with_commas[0:next_comma])
                # print("parse rest: ", rest)
                return [parse_tag_value(list_str_with_commas[0:next_comma])] + parse_tag_value("[" + rest + "]")
            else:
                return [parse_tag_value(list_str_with_commas)]
        elif isinstance(tag_value, str) and tag_value.startswith("'") and tag_value.endswith("'"):
            # remove quotes
            return parse_tag_value(tag_value[1:-1])

        elif isinstance(tag_value, str) and tag_value.startswith('"') and tag_value.endswith('"'):
            # remove quotes
            return parse_tag_value(tag_value[1:-1])

        else:
            try:
                int_value = int(tag_value)
                # print("parsed int: ", int_value)
                return int_value
            except ValueError:
                pass
            try:
                float_value = float(tag_value)
                # print("parsed float: ", float_value)
                return float_value
            except ValueError:
                pass

    # print("returning as is: ", tag_value)
    return tag_value


def compute_orientation_from_agg_tag_value(agg_tag_value: str) -> str:
    if isinstance(agg_tag_value, str):
        if agg_tag_value.startswith("n="):
            # Extract the number of orientations from the string
            num_orientations = int(agg_tag_value[2:].split("/")[0])
            num_slices = int(agg_tag_value[2:].split("/")[1])
            if num_orientations < 8:
                return "ORTHO"
            elif num_orientations > 8 and num_slices > 1:
                return "ROT"
        else:
            return compute_orientation(parse_tag_value(agg_tag_value))

    elif isinstance(agg_tag_value, List):
        return compute_orientation(agg_tag_value)
    elif isinstance(agg_tag_value, float) and np.isnan(agg_tag_value):
        return "na"
    else:
        print("Unexpected type for agg_tag_value:", type(agg_tag_value))
        print("Value:", agg_tag_value)

    return "na"


def encode_image_orientation_patient(dicom_tags_df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the 'ImageOrientationPatient' column of the provided DICOM tags
    DataFrame.

    Parameters:
        dicom_tags_df (pd.DataFrame): Input DataFrame containing DICOM tags.
            Must be a pandas DataFrame with an 'ImageOrientationPatient' column.
    Returns:
        pd.DataFrame: DataFrame with encoded ImageOrientationPatient.
    """
    if "ImageOrientationPatient" not in dicom_tags_df.columns:
        dicom_tags_df.insert(column="ImageOrientationPatient", value=np.nan, loc=len(dicom_tags_df.columns))

    acquisition_plane = dicom_tags_df["ImageOrientationPatient"].apply(compute_orientation_from_agg_tag_value)
    # Create a new column for encoded ImageOrientationPatient
    dicom_tags_df["enc_AX_ImageOrientationPatient"] = acquisition_plane.apply(lambda x: 1 if x == "AX" else 0)
    dicom_tags_df["enc_SAG_ImageOrientationPatient"] = acquisition_plane.apply(lambda x: 1 if x == "SAG" else 0)
    dicom_tags_df["enc_COR_ImageOrientationPatient"] = acquisition_plane.apply(lambda x: 1 if x == "COR" else 0)
    dicom_tags_df["enc_ORTHO_ImageOrientationPatient"] = acquisition_plane.apply(lambda x: 1 if x == "ORTHO" else 0)
    dicom_tags_df["enc_ROT_ImageOrientationPatient"] = acquisition_plane.apply(lambda x: 1 if x == "ROT" else 0)

    # overwrite with 2, if input was nan
    dicom_tags_df.loc[dicom_tags_df["ImageOrientationPatient"] == "NaN", "enc_AX_ImageOrientationPatient"] = 2
    dicom_tags_df.loc[dicom_tags_df["ImageOrientationPatient"] == "NaN", "enc_SAG_ImageOrientationPatient"] = 2
    dicom_tags_df.loc[dicom_tags_df["ImageOrientationPatient"] == "NaN", "enc_COR_ImageOrientationPatient"] = 2
    dicom_tags_df.loc[dicom_tags_df["ImageOrientationPatient"] == "NaN", "enc_ORTHO_ImageOrientationPatient"] = 2
    dicom_tags_df.loc[dicom_tags_df["ImageOrientationPatient"] == "NaN", "enc_ROT_ImageOrientationPatient"] = 2

    # drop the original column
    return dicom_tags_df.drop(columns=["ImageOrientationPatient"])


def parse_pixel_spacing(x: Any) -> List[float]:
    """Parse pixel spacing from a string or list and convert to float.

    Args:
        x (str or list): Pixel spacing value as a string or list of strings.
    Returns:
        list: List of pixel spacing values as floats.
    """

    if isinstance(x, list):
        return [float(i) for i in x]
    if isinstance(x, str):
        try:
            vals = literal_eval(x)
            if isinstance(vals, (list, tuple)):
                return [float(i) for i in vals]
        except Exception:
            pass
    return []


def encode_pixel_spacing(dicom_tags_df: pd.DataFrame) -> pd.DataFrame:
    if "PixelSpacing" not in dicom_tags_df.columns:
        dicom_tags_df.insert(column="PixelSpacing", value=np.nan, loc=len(dicom_tags_df.columns))

    # Convert "PixelSpacing" column to list of floats
    dicom_tags_df["PixelSpacing"] = dicom_tags_df["PixelSpacing"].apply(parse_pixel_spacing)

    # split PixelSpacing into two separate columns
    dicom_tags_df["PixelSpacing_y"] = dicom_tags_df["PixelSpacing"].apply(lambda x: x[1] if len(x) > 1 else None)
    dicom_tags_df["PixelSpacing_x"] = dicom_tags_df["PixelSpacing"].apply(lambda x: x[0] if len(x) > 0 else None)

    # overwrite with 2, if input was nan
    dicom_tags_df.loc[dicom_tags_df["PixelSpacing"].isna(), "PixelSpacing_x"] = np.nan
    dicom_tags_df.loc[dicom_tags_df["PixelSpacing"].isna(), "PixelSpacing_y"] = np.nan

    return dicom_tags_df.drop(columns=["PixelSpacing"])


def encode_labels(labels_df: pd.DataFrame, label_names: Dict[str, List[str]]) -> pd.DataFrame:
    """Encodes the labels in the provided DataFrame.

    Args:
        labels_df (pd.DataFrame): DataFrame containing labels.
        label_names (Dict[str, List[str]]): Dictionary with label columns to encode.
    Returns:
        pd.DataFrame: DataFrame with each value in label_names encoded as binary feature.
    """
    encoding_cols = []

    # insert missing columns from label_names with "na" values
    for class_name in label_names.keys():
        if class_name not in labels_df.columns:
            labels_df.insert(column=class_name, value="na", loc=len(labels_df.columns))

    # reorder columns to consistent order as in label_names, drop anything else
    labels_df = labels_df.loc[:, [col for col in label_names.keys()]]

    for class_name, values in label_names.items():
        # add encoding column
        labels_df.insert(loc=len(labels_df.columns), column=f"enc_{class_name}", value=np.nan)
        encoding_cols.append(f"enc_{class_name}")

        for val_idx, val in enumerate(values):
            if val != "na":
                # Create a new column encoding for each label value
                labels_df.loc[labels_df[class_name] == val, f"enc_{class_name}"] = val_idx
            else:
                # This shouldn't happen in labels should it?
                labels_df.loc[labels_df[class_name] == "na", f"enc_{class_name}"] = np.nan

    return labels_df.drop(columns=[col for col in labels_df.columns if col not in encoding_cols])


def check_alignment(
    df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str, column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Check if two DataFrames are aligned by a specific column.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame to check.
    df2 : pd.DataFrame
        Second DataFrame to check.
    column : str
        Column name to check alignment on.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Aligned DataFrames.
    """
    # check if both DataFrames have the column:
    if column is not None:
        if column not in df1.columns:
            raise ValueError(f"Column '{column}' not found in {name1}. ")
        if column not in df2.columns:
            raise ValueError(f"Column '{column}' not found in {name2}. ")

    if len(df1) != len(df2):
        log.warning(
            f"df1 and df2 do not have the same number of rows. Got len({name1}): {len(df1)}, len({name2}): {len(df2)}."
        )
        len_df1 = len(df1)
        df1.drop_duplicates(inplace=True)
        if len_df1 != len(df1):
            log.warning(f"Dropped {len_df1 - len(df1)} duplicate rows in {name1}.")
        len_df2 = len(df2)
        df2.drop_duplicates(inplace=True)
        if len_df2 != len(df2):
            log.warning(f"Dropped {len_df2 - len(df2)} duplicate rows in {name2}.")

        if column is not None:
            df1 = df1.loc[df1[column].isin(df2[column]), :]
            df2 = df2.loc[df2[column].isin(df1[column]), :]
            if len(df1) != len(df2):
                log.warning(
                    f"After aligning by {column}, lengths are still different. Got len({name1}): {len(df1)}, len({name2}): {len(df2)}."
                )
                raise ValueError(f"Could not match DataFrames {name1} and {name2} by column {column}. ")
            else:
                log.warning(f"df1 and df2 were aligned by column {column} and thereby reduced to length {len(df1)}.")

                # sort by given column::
                df1 = df1.set_index(column, drop=False).sort_index()
                df2 = df2.set_index(column, drop=False).sort_index()

                if not df1[column].equals(df2[column]):
                    raise ValueError(f"{column} in {name1} and {name2} still do not match. ")
                # restore column Filepath:
                df2 = df2.reset_index(drop=True)
                df1 = df1.reset_index(drop=True)
        elif len(df1) != len(df2):
            raise ValueError(
                f"df1 and df2 do not have the same number of rows after dropping duplicates. Got len({name1}): {len(df1)}, len({name2}): {len(df2)}."
            )
    elif column is not None:
        if not df1[column].equals(df2[column]):
            log.info(f"{column} in {name1} and {name2} do not match. Resorting and checking again...")

            # sort by given column::
            df1 = df1.set_index(column, drop=False).sort_index()
            df2 = df2.set_index(column, drop=False).sort_index()

            if not df1[column].equals(df2[column]):
                raise ValueError(f"{column} in {name1} and {name2} still do not match. ")
            else:
                log.info(f"df1 and df2 were aligned by column {column} by sorting.")

            # restore column Filepath:
            df2 = df2.reset_index(drop=True)
            df1 = df1.reset_index(drop=True)

    elif not df2.index.equals(df1.index):
        log.info(f"Indices in {name1} and {name2} do not match. Ignoring indices and resetting them.")
        # drop indices:
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

    return df1, df2


def encode_dicom_tags_by_version(dicom_tags_df: pd.DataFrame, dicom_encoding_version: str = "brain") -> pd.DataFrame:
    """Select and encode a predefined set of DICOM tags from the provided
    DataFrame.

    Parameters
    ----------
    dicom_tags_df : pd.DataFrame
        DataFrame containing DICOM tags.
        Example columns:
        "Filepath", "AngioFlag", "Columns", "DiffusionBValue", ...
    selected_tags_version : str, optional
        Version of selected tags to use, by default "brain".
        Currently only "brain" is implemented.

    Returns
    -------
    pd.DataFrame
        DataFrame with selected and encoded DICOM tags.
    """
    common_categorical_tags = [
        "AngioFlag",
        "ImageOrientationPatient",
        "ImageType",
        "MRAcquisitionType",
        "ScanningSequence",
        "ScanOptions",
        "SeriesDescription",
        "SequenceVariant",
    ]

    common_numerical_tags = [
        # "AcquisitionTime",
        "AcquisitionDuration",
        "Columns",  # number of columns in the image
        "DiffusionBValue",  # b-value for diffusion images
        "EchoNumbers",
        "EchoTime",
        "EchoTrainLength",
        "FlipAngle",
        "ImagesInSeries",
        "MagneticFieldStrength",
        "NumberOfAverages",
        "PercentPhaseFieldOfView",
        "PercentSampling",
        "PixelBandwidth",
        "PixelSpacing",
        "RepetitionTime",
        "Rows",
        "SamplesPerPixel",
        "SliceThickness",
    ]
    encoded_dicom_tags_df = dicom_tags_df.copy()

    # select dicom tags according to version:
    if dicom_encoding_version == "brain":
        categorical_tags = common_categorical_tags
        numerical_tags = common_numerical_tags

    ###
    # add further encoding versions here
    ###

    else:
        raise ValueError(f"Unknown dicom_encoding_version: {dicom_encoding_version}")

    # add np.nan as missing values for missing tags:
    for tag in categorical_tags + numerical_tags:
        if tag not in encoded_dicom_tags_df.columns:
            encoded_dicom_tags_df.insert(column=tag, value=np.nan, loc=len(encoded_dicom_tags_df.columns))
            log.info(f"Missing DICOM tag {tag} added with NaN values.")

    added_features = []
    # add indicator features, if dicom tags is nan
    for tag in numerical_tags:
        encoded_dicom_tags_df.insert(
            column=f"{tag}_missing",
            value=encoded_dicom_tags_df[tag].isna().astype(int),
            loc=len(encoded_dicom_tags_df.columns),
        )
    added_features.append(f"{tag}_missing")

    # add indicator features, if dicom tags start with "n="
    for tag in categorical_tags + numerical_tags:
        encoded_dicom_tags_df.insert(
            column=f"{tag}_multiple",
            value=encoded_dicom_tags_df[tag].astype(str).str.startswith("n=").astype(int),
            loc=len(encoded_dicom_tags_df.columns),
        )
    added_features.append(tag + "_multiple")

    # encode categorical dicom tags into binary features according to version:
    if dicom_encoding_version == "brain":
        encoded_dicom_tags_df = encode_angio_flag(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_series_description(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_image_orientation_patient(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_image_type(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_scan_option(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_scanning_sequence(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_mr_acquisition_type(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_sequence_variant(encoded_dicom_tags_df)
        encoded_dicom_tags_df = encode_pixel_spacing(encoded_dicom_tags_df)
    else:
        raise ValueError(f"Unknown dicom_encoding_version: {dicom_encoding_version}")

    # drop all other columns:
    all_columns = ["Filepath"] + categorical_tags + numerical_tags + added_features

    tags_to_drop = [col for col in encoded_dicom_tags_df.columns if col not in all_columns]
    encoded_dicom_tags_df = encoded_dicom_tags_df.drop(columns=tags_to_drop, errors="ignore")

    encoded_dicom_tags_df = encoded_dicom_tags_df.reindex(columns=sorted(all_columns))

    # make sure all values are numeric now:
    for col in encoded_dicom_tags_df.columns:
        if col != "Filepath":  # keep Filepath as string
            # convert to numeric, coerce errors to NaN
            encoded_dicom_tags_df[col] = pd.to_numeric(encoded_dicom_tags_df[col], errors="coerce")

    return encoded_dicom_tags_df


def encode_dicom_and_pixel_features(
    dicom_tags_df: pd.DataFrame,
    pixel_preds_df: pd.DataFrame,
    label_names: Dict[str, List[str]],
    dicom_encoding_version: str = "brain",
) -> pd.DataFrame:
    """Prepare the input DataFrame for the Random Forest Classifier by merging
    and encoding a DataFrames with DICOM tags and DataFrame with pixel model
    prediction scores.

    This method assumes, that indices of dicom_tags_df and pixel_preds_df are aligned,
    if both are not empty!

    Parameters
    ----------
    dicom_tags_df : pd.DataFrame
        DataFrame containing DICOM tags.
        Example columns:
        "Filepath", "AngioFlag", "Columns", "DiffusionBValue", ...
    pixel_preds_df : pd.DataFrame
        DataFrame containing pixel model prediction probabilities.
        Example columns:
        "Filepath", "pred_label_AcquisitionPlane_prob_AX", "pred_label_AcquisitionPlane_prob_COR", ...
    dicom_encoding_version : str, optional
        Version of selected tags to use, by default "brain".

    Returns
    -------
    pd.DataFrame
        Merged encodings ready to feed into Random Forest
    """
    encoded_dicom_tags_df = encode_dicom_tags_by_version(dicom_tags_df, dicom_encoding_version)

    encoded_pixel_preds_df = pixel_preds_df.copy()
    # if there is no pixel model prediction input, ignore it:
    if not encoded_pixel_preds_df.empty:
        for label_class, label_values in label_names.items():
            for label_value in label_values:
                # encode label values with indices:
                encoded_pixel_preds_df.loc[
                    encoded_pixel_preds_df[label_class.replace("label_", "pred_")] == label_value,
                    label_class.replace("label_", "pred_"),
                ] = label_values.index(label_value)

                # check if all probabilities are there
                prob_feature_name = label_class.replace("label_", "pred_") + "_prob_" + label_value
                if prob_feature_name not in encoded_pixel_preds_df.columns:
                    encoded_pixel_preds_df.insert(
                        column=prob_feature_name, value=0.0, loc=len(encoded_pixel_preds_df.columns)
                    )
                    log.warning(
                        f"Missing pixel model prediction probabilities for {label_class} value {label_value} added with 0.0 values."
                    )

            # normalize probabilities to 0...1 for each label_class
            probs_this_class = [
                label_class.replace("label_", "pred_") + "_prob_" + label_value for label_value in label_values
            ]
            encoded_pixel_preds_df.loc[:, probs_this_class].apply(
                lambda x: (x - encoded_pixel_preds_df[probs_this_class].min(axis=1))
                / (
                    encoded_pixel_preds_df[probs_this_class].max(axis=1)
                    - encoded_pixel_preds_df[probs_this_class].min(axis=1)
                )
            )

        # remove any other columns that are not Filepath, prediction, or prediction probabilities:
        remove_cols = []
        for col in encoded_pixel_preds_df.columns:
            if col != "Filepath" and not col.startswith("pred_"):
                remove_cols.append(col)
        if len(remove_cols) > 0:
            encoded_pixel_preds_df.drop(columns=remove_cols, errors="ignore", inplace=True)

        # sort columns:
        encoded_pixel_preds_df.sort_index(axis=1, inplace=True)

    # ensure encoded_dicom_tags_df and encoded_pixel_preds_df are aligned:
    if len(encoded_dicom_tags_df) > 0 and len(encoded_pixel_preds_df) > 0:
        encoded_dicom_tags_df, encoded_pixel_preds_df = check_alignment(
            encoded_dicom_tags_df,
            encoded_pixel_preds_df,
            "encoded_dicom_tags_df",
            "encoded_pixel_preds_df",
            column="Filepath",
        )

    # merge columns and remove "Filepath"
    input_df = pd.concat([encoded_dicom_tags_df, encoded_pixel_preds_df], axis=1, ignore_index=False).drop(
        columns=["Filepath"]
    )

    return input_df





class LiverDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 100,
        n_slices: int = 3,
        img_channels: int = 1,
        img_size: int = 224,
        metadata_dim: int = 512,
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
        label_path: str = "/home/melanie.dohmen/iml-series-labelling/data/PVai_full/labels_20250603.csv",
        augment_conf: str = "NONE2D",
    ):
        self.num_samples = num_samples
        self.n_slices = n_slices
        self.img_channels = img_channels
        self.img_size = img_size
        self.metadata_dim = metadata_dim
        self.label_names = label_names
        self.augment_conf = augment_conf

        labels_df = pd.read_csv(label_path)

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

        labels_df = labels_df.set_index("Filepath")

        for index, row in labels_df.iterrows():
            # add file path to path list
            self.path_list.append(index)
            # get label idx:
            self.labels.append(row.to_dict())

        print("len(self.path_list) =", len(self.path_list))

    def __len__(self: Any) -> int:
        return self.num_samples

    
    def get_n_labels(self):
        r = {}
        for x in self.label_names:
            n = len(self.label_names[x])
            r[x] = n
        return r

    def get_cache_size(self):
        sz = len(self.img_buffer)
        return sz
        
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
        
        # sample distinct random slices
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
                
            # key for buffering
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
                self.img_buffer[fid] = dcm_image.pixel_array.astype(np.float32)
       
            slice_images.append(copy.deepcopy(self.img_buffer[fid]))
        
        return slice_images
        

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        # --- IMAGES ---
        image_list = []
        
        dcm_images = self.open_dicom_slice_from_series(self.path_list[idx], sampling_type="equidistant", stop_before_pixels=False, n_images = self.n_slices)

        for image in dcm_images:
            
            # we dont accept images with channel dimension
            if (image is None) or (len(image.shape)>2):
                 image = np.zeros((244, 244), dtype=np.float32)
                
            # in dicom series, channel dimension is first, so we need to transpose it
            # TODO: When does this happen ? We want GV images only
            #if len(image.shape) == 3:
            #    log.warning(f"Multi-channel image detected (shape {image.shape})! File: {self.path_list[idx]}")
            #    if  image.shape[0] == 1:
            #        image = image[0]
            #    else:
            #        image = np.zeros(244,244)

            # data augmentation
            image = augment(image, self.augment_conf)

            #image_list.append(torch.Tensor(np.stack([image, image, image], axis=0)).to(torch.float32))
            image_list.append(torch.Tensor(image).unsqueeze(0).to(torch.float32))
            
        images = torch.stack(image_list, dim=0)  # (N_slices, C, H, W)

        # --- METADATA ---
        metadata = torch.randn(self.metadata_dim)

        # --- TARGETS ---
        label_idx_dict: Dict[str, int] = {}

        for label_class, label_value in self.labels[idx].items():
            if label_class in self.label_names:
                if label_value in self.label_names[label_class]:
                    label_idx_dict[label_class] = self.label_names[label_class].index(label_value)
                else:
                    label_idx_dict[label_class] = -1
                    log.warning(
                        f"Label value {label_value} not in label names of label class {label_class}, found {self.label_names[label_class]} only"
                    )
            else:
                # label_idx_dict[label_class] = -1
                log.debug(f"Label class {label_class} not in label names, found {self.label_names.keys()} only.")

        targets = tuple(torch.tensor(label_idx_dict.get(label_class, -1)) for label_class in self.label_names.keys())

        return images, metadata, targets


# Usage example:
if __name__ == "__main__":
    dummy_dataset = LiverDataset(num_samples=200, n_slices=5, metadata_dim=3 * 256, label_path="~/pvai_labels_20250603.csv")
    dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

    n_label = dummy_dataset.get_n_labels()
    print(n_label)
    
    for batch_idx, (images, metadata, targets) in enumerate(dummy_loader):
        print(f"Batch {batch_idx}:")
        print(f"  images.shape = {images.shape}")  # (B, N_slices, C, H, W)
        print(f"  metadata.shape = {metadata.shape}")  # (B, metadata_dim)
        print(f"  targets shapes = {[t.shape for t in targets]}")

        print(targets)
        if batch_idx == 1:  # just show first 2 batches
            break
