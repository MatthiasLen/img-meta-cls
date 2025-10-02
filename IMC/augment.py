"""
Author:  Melanie Dohmen
Version: 2025-09-29
"""


import logging
import elasticdeform
import numpy as np

from skimage.filters import gaussian

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

VALID_CONFIGURATIONS = {
    "DEFAULT2D": DEFAULT2D,
    "ZSCOREDEFAULT2D": ZSCOREDEFAULT2D,
    "NONE2D": NONE2D,
    "ZSCORENONE2D": ZSCORENONE2D
}


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

    augment_dict = VALID_CONFIGURATIONS[augment_conf] if augment_conf in VALID_CONFIGURATIONS else NONE2D

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
