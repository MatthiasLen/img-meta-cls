"""DICOM metadata encoder — version 2.

This module reads raw DICOM tags from DICOM files, aggregates them per series,
and encodes them into a flat numerical feature dictionary (``Dict[str, Any]``).

The resulting feature dictionaries can be used as input to machine-learning
models (e.g., Random Forest or neural network metadata encoder).

Typical usage::

    from IMC.data.dicom_tag_encoding_v2 import generate_slicewise_metadata_vector

    encoded_vectors, raw_tags = generate_slicewise_metadata_vector(slice_filenames)
    # encoded_vectors is a List[Dict[str, Any]] with one dict per slice

For batch processing and writing results to a Parquet file, use
:mod:`IMC.data.encode_metadata`.
"""

import logging
import os
from ast import literal_eval
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import numpy as np

# only needed for aggregation
import pandas as pd
from pydicom import dcmread

log = logging.getLogger(__name__)


# DICOM tags names (following the DICOM standard)
# to be extracted for metadata encoding
# and their expected types/format
SELECTED_RAW_DICOM_TAGS = {
    "AcquisitionDuration": float,
    "AngioFlag": str,
    "Columns": int,
    "ContrastBolusAgent": str,
    "ContrastBolusVolume": float,
    "DiffusionBValue": float,
    "EchoNumbers": int,
    "EchoTime": float,
    "EchoTrainLength": int,
    "FlipAngle": float,
    "ImageOrientationPatient": List[float],
    "ImageType": List[str],
    "MagneticFieldStrength": float,
    "Modality": str,
    "MRAcquisitionType": str,
    "NumberOfAverages": float,
    "PercentPhaseFieldOfView": float,
    "PercentSampling": float,
    "PixelBandwidth": float,
    "PixelSpacing": List[float],
    "RepetitionTime": float,
    "Rows": int,
    "SamplesPerPixel": int,
    "SliceLocation": float,
    "SliceThickness": float,
    "ScanningSequence": List[str],
    "ScanOptions": List[str],
    "SeriesDescription": str,
    "SequenceVariant": List[str],
}


def read_dicom_tags(slice_path: str) -> Dict[str, Any]:
    """Read DICOM tags from a DICOM slice file into a dictionary and add
    filepath and series path.

    Args:
        slice_path (str): Path to the DICOM slice file.

    Returns:
        Dict[str, Any]: Dictionary containing the extracted DICOM tags and their values.
    """

    ds = dcmread(slice_path, stop_before_pixels=True)
    dicom_tags = {tag: ds.data_element(tag).value for tag in ds.dir()}

    return dicom_tags


def check_dicom_tag_types(dicom_tags: Dict[str, Any], selected_dicom_tags: Dict[str, type]) -> Dict[str, Any]:
    """Check and convert DICOM tag types to expected types.

    Args:
        dicom_tags (Dict[str, Any]): Dictionary containing DICOM tags and their values.
        selected_dicom_tags (Dict[str, type]): Dictionary specifying expected types for selected DICOM tags.
    Returns:
        Dict[str, Any]: Dictionary with selected DICOM tags converted to expected types.
    """

    dicom_tags = {tag_name: tag_value for tag_name, tag_value in dicom_tags.items() if tag_name in selected_dicom_tags}

    for tag_name, tag_type in selected_dicom_tags.items():
        if tag_name in dicom_tags and dicom_tags[tag_name] is not None:
            if tag_type == List[float]:
                try:
                    dicom_tags[tag_name] = [float(x) for x in dicom_tags[tag_name]]
                except Exception:
                    log.warning(
                        f"DICOM tag {tag_name} with value {dicom_tags[tag_name]} could not be converted to expected type {tag_type}"
                    )
                    dicom_tags[tag_name] = None  # or some default value
            elif tag_type == List[str]:
                try:
                    if isinstance(literal_eval(str(dicom_tags[tag_name])), list):
                        dicom_tags[tag_name] = [str(x) for x in literal_eval(str(dicom_tags[tag_name]))]
                    else:
                        dicom_tags[tag_name] = [str(dicom_tags[tag_name])]
                except ValueError:
                    if isinstance(dicom_tags[tag_name], str):
                        dicom_tags[tag_name] = [dicom_tags[tag_name]]
                    else:
                        log.warning(
                            f"DICOM tag {tag_name} with value {dicom_tags[tag_name]} could not be converted to expected type {tag_type}"
                        )
                        dicom_tags[tag_name] = None  # or some default value
                except TypeError:
                    log.warning(
                        f"DICOM tag {tag_name} with value {dicom_tags[tag_name]} could not be converted to expected type {tag_type}"
                    )
                    dicom_tags[tag_name] = None  # or some default value

                except Exception:
                    log.warning(
                        f"DICOM tag {tag_name} with value {dicom_tags[tag_name]} could not be converted to expected type {tag_type}"
                    )
                    dicom_tags[tag_name] = None  # or some default value

            else:  # convert to str, float or int
                if tag_name in dicom_tags:
                    try:
                        dicom_tags[tag_name] = tag_type(dicom_tags[tag_name])
                    except TypeError:
                        log.warning(
                            f"DICOM tag {tag_name} with value {dicom_tags[tag_name]} could not be converted to expected type {tag_type}"
                        )
                        dicom_tags[tag_name] = None  # or some default value
                    except Exception:
                        log.warning(
                            f"DICOM tag {tag_name} with value {dicom_tags[tag_name]} could not be converted to expected type {tag_type}"
                        )
                        dicom_tags[tag_name] = None  # or some default value
        else:
            dicom_tags[tag_name] = None  # or some default value

    return dicom_tags


def compute_orientation(orientation_vector: Union[List[float], None]) -> Union[str, None]:
    """Compute orientation of dicom image based on ImageOrientationPatient tag.

    Args:
        orientation_vector (List[float]): List of 6 float values representing ImageOrientationPatient tag.
    Returns:
        str: string indicating acquisition plane SAG -> sagittal, COR->coronal, AX-> axial, NA if image_ori not available
    """

    if orientation_vector is not None:
        try:
            orientation_vector = [float(x) for x in orientation_vector]
        except (ValueError, TypeError):
            log.warning(
                f"ImageOrientationPatient tag has invalid values: {orientation_vector}. Cannot compute orientation."
            )
            return None

        if len(orientation_vector) != 6:
            log.warning(
                f"ImageOrientationPatient tag has invalid length: {len(orientation_vector)}. Expected length is 6."
            )
            return None

        orientation = np.array(orientation_vector)
        # vector components along y axis [yx, yy, yz]
        image_y = orientation[0:3]
        # vector components along x axis [xx, xy, xz]
        image_x = orientation[3:6]

        # compute projection along z axis
        image_z = np.cross(image_x, image_y)
        abs_image_z = abs(image_z)

        # find closest unit vector to the nomal
        main_index = list(abs_image_z).index(max(abs_image_z))

        if main_index == 0:
            return "SAG"
        elif main_index == 1:
            return "COR"
        elif main_index == 2:
            return "AX"

    return None


def aggregate_dicom_tags(dicom_tags_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DICOM tags for all slices belonging to a DICOM series.

    Most tags are aggregated by keeping the single unique value, or
    ``"n=<count>"`` when multiple unique values exist.

    Exceptions:

    - ``ImagesInSeries``: set to the total number of slices.
    - ``SeriesInstanceUID`` and ``SeriesDescription``: first value is kept
      with a ``"MultiN WithFirst=<value>"`` prefix when duplicates exist.
    - ``ImageOrientationPatient`` / ``AcquisitionPlane``: returns
      ``"ORTHO"`` when multiple orthogonal planes are detected, and
      ``"ROT"`` when the number of unique orientation vectors matches the
      number of slices (rotating acquisition).

    Args:
        dicom_tags_df: DataFrame containing DICOM tags for all slices in a
            series.  The index must be named ``"Filepath"`` and a
            ``"SeriesFilepath"`` column must be present (used as the index
            of the returned DataFrame).  Typical columns include
            ``"AngioFlag"``, ``"Columns"``, ``"DiffusionBValue"``, …

    Returns:
        Single-row DataFrame with aggregated DICOM tags for the series.
    """

    if "ImagesInSeries" not in dicom_tags_df.columns:
        dicom_tags_df.insert(loc=1, column="ImagesInSeries", value=0)
    if "ImageOrientationPatient" not in dicom_tags_df.columns:
        dicom_tags_df.insert(loc=1, column="ImageOrientationPatient", value=None)
    if "AcquisitionPlane" not in dicom_tags_df.columns:
        dicom_tags_df.insert(loc=1, column="AcquisitionPlane", value=None)

    # compute AcquisitionPlane from ImageOrientationPatient,
    # because aggregation should be done on derived acquisition planes
    dicom_tags_df.loc[:, "AcquisitionPlane"] = dicom_tags_df["ImageOrientationPatient"].apply(compute_orientation)

    def unique_values(x: pd.Series) -> Any:
        nunique = x.astype(str).nunique()
        # include the number of slices in the series
        if x.name == "ImagesInSeries":
            return len(x)
        # if only one unique value, return that value
        elif nunique == 1:
            return x.iloc[0]
        # for these tags, return first value with indication of multiple unique values
        elif x.name in ["SeriesInstanceUID", "SeriesDescription"]:
            return "Multi" + str(nunique) + "WithFirst=" + str(x.iloc[0])
        # for AcquisitionPlane, check for orthogonal planes
        elif x.name == "AcquisitionPlane":
            if nunique == 1:
                return x.iloc[0]
            elif None not in x.values:
                return "ORTHO"
            else:
                return None
        # else return number of unique values:
        else:
            return "n=" + str(nunique)

    # select unique value or count unique values if more than one
    # transpose output series to a DataFrame with 1 row
    agg_dicom_tags_df = pd.DataFrame(dicom_tags_df.agg(lambda x: unique_values(x))).transpose()

    # adjust AcquisitionPlane in aggretation for rotating acqusition planes:
    # There is only one row left after aggregation, therefore the aggregated values can be accessed directly via .iloc[0]
    # check:
    # if ImageOrientationPatient has multiple unique values
    # if there were multiple AcquisitionPlanes (were aggregated as ORTHO)
    # if number of unique ImageOrientationPatient vectors equals number of slices (ImagesInSeries)
    # and there were at least 6 slices in the series
    try:
        if (
            ("ImageOrientationPatient" in agg_dicom_tags_df.columns)
            and (str(agg_dicom_tags_df.iloc[0]["ImageOrientationPatient"]).startswith("n="))
            and (agg_dicom_tags_df.iloc[0]["AcquisitionPlane"] == "ORTHO")
            and (
                int(agg_dicom_tags_df.iloc[0]["ImageOrientationPatient"][2:])
                == agg_dicom_tags_df.iloc[0]["ImagesInSeries"]
            )
            and (agg_dicom_tags_df.iloc[0]["ImagesInSeries"] >= 6)
        ):
            agg_dicom_tags_df.loc[:, "AcquisitionPlane"] = "ROT"
    except Exception as e:
        log.warning(f"Could not adjust AcquisitionPlane for rotating planes: {e}")

    return agg_dicom_tags_df


def encode_nonfixed_category(
    input_value: Union[List[str], str, None], name: str, values: Dict[str, List[str]]
) -> Dict[str, float]:
    """Encodes a categorical value according to a non-fixed set of possible
    values that could be found in the input.

    Parameters:
        input_value (list or str or None): Value(s) to encode.
        name (str): Name of the category to encode.
        values (dict): keys are names of groups to encode, values are lists of possible values to search for in the input_value.

    Returns:
        Dict[str, float]:
            Contains a key'enc_{name}_{key}' for each key in the values dictionary

    Example:
        1) Given input_value (List[str]) = ["DERIVED", "PRIMARY", "MIP"], name = "ImageType" and
        values = {
            "projection": ["PROJECTION", "MIP"],
            "derived": ["DERIVED"],
            "diffusion": ["DIFFUSION"],
        }

        The function will return a recarray with three fields:
            enc_{name}_projection: 1 (since "MIP" is in the input_value)
            enc_{name}_derived: 1 (since "DERIVED" is in the input_value)
            enc_{name}_diffusion: 0 (since "DIFFUSION" is not in the input_value)

        2) Given input_value (str) = "Localizer Scan", name = "SeriesDescription" and
        values = {
            "loc": ["loc", "scout", "survey"],
            "mrcp": ["mrcp"],
        }

        The function will return a recarray with two fields:
            enc_{name}_loc: 1 (since "loc" is in the input_value)
            enc_{name}_mrcp: 0 (since "mrcp" is not in the input_value)

        Attention: If input_value is str, comparison is done case insensitive.
    """
    if input_value is not None:
        try:
            # do case insensitive comparison for str input values
            if isinstance(input_value, str):
                input_value = input_value.lower()

            # check if one of the possible values is found in the input_value list
            return {f"enc_{name}_{key}": int(any(val in input_value for val in values[key])) for key in values}

        except TypeError:
            log.warning(
                f"Input value: {input_value} for DICOM Tag {name} is not iterable, but should be of type {SELECTED_RAW_DICOM_TAGS[name]}. Encoding as missing."
            )

    # in case of None value or error, encode as missing
    return {f"enc_{name}_{key}": 2 for key in values}


def encode_fixed_category(input_value: Union[str, None], name: str, values: List[str]) -> Dict[str, float]:
    """Encode a categorical value against a fixed set of possible values.

    Parameters:
        input_value: Value to encode.  ``None`` is treated as missing.
        name: Tag name used to construct output keys (``enc_{name}_{val}``).
        values: Exhaustive list of expected category values.

    Returns:
        Dict with a key ``enc_{name}_{val}`` for each element of *values*:
        ``1.0`` for the matching value, ``0.0`` for all others, or ``2.0``
        for every key when *input_value* is ``None`` or not found in *values*.
    """

    if input_value is not None:
        # check if input_value matches one of the possible values
        if not any(input_value == val for val in values):
            log.warning(
                f"Input value: {input_value} for DICOM Tag {name} not found in expected values {values}. Encoding as missing."
            )
            return {f"enc_{name}_{val}": 2 for val in values}
        else:
            # encodes as 1, if input_value matches a certain value in the given list, else 0
            return {f"enc_{name}_{val}": int(input_value == val) for val in values}
    else:
        return {f"enc_{name}_{val}": 2 for val in values}


def encode_dicom_tags_by_version(
    dicom_tags_dict: Dict[str, Dict[str, Any]], encoding_version: str = "version_1"
) -> Dict[str, Dict[str, Any]]:
    if encoding_version == "version_1":
        encoded_dicom_tags_dict: Dict[str, Dict[str, Any]] = {}
        # iterate over each row in dicom_tags_df, which corresponds to one aggredated series or one slice
        for filepath, dicom_tags in dicom_tags_dict.items():
            encoded_tags_per_filepath = encode_dicom_tags_version_1(dicom_tags)
            encoded_dicom_tags_dict[filepath] = encoded_tags_per_filepath
        return encoded_dicom_tags_dict

    # elif encoding_version == "version_X":
    #     add further encoding versions here

    else:
        raise NotImplementedError(f"Encoding version '{encoding_version}' is not implemented.")


def encode_dicom_tags_version_1(dicom_tags: Dict[str, Any]) -> Dict[str, Any]:
    """Encode a dictionary of DICOM tags (aggregated DICOM tags of one series
    or DICOM tags of one slice) into a numerical feature dictionary.

    In this version:
      - encode missing values for categorical tags as value 2
      - encode missing values for numerical tags as additional binary feature {tag}_missing
      - encode number of unique values for categorical and numerical tags as additional feature {tag}_multiple
      - special handling of PixelSpacing tag: encode x and y pixel spacing as separate numerical features
      - special handling of ImageOrientationPatient/AcquisitionPlane tag: compute from ImageOrientationPatient, or use aggregated values (ROT, ORTHO)
      - special handling of ContrastBolusAgent tag: encode whether empty string is present
      - add enc_ImagesInSeries from aggregation as additional feature

    Args:
        dicom_tags: Dictionary containing DICOM tags.

    Returns:
        Encoded DICOM tags as a plain ``Dict[str, Any]`` with sorted keys.
    """

    # define dicom tags, where features are encoded in a special way
    special_tags = ["ImageOrientationPatient", "PixelSpacing", "ContrastBolusAgent"]

    # encode selected dicom tags with type str or List[str] as categorical tags:
    categorical_tags = [
        tag_name
        for tag_name in SELECTED_RAW_DICOM_TAGS
        if SELECTED_RAW_DICOM_TAGS[tag_name] in [str, List[str]] and tag_name not in special_tags
    ]
    # encode selected dicom tags with type int or float as numerical tags:
    numerical_tags = [
        tag_name for tag_name in SELECTED_RAW_DICOM_TAGS if SELECTED_RAW_DICOM_TAGS[tag_name] in [int, float]
    ]

    # add None value for tags, that are not present in the input dicom_tags dict
    for tag in categorical_tags + numerical_tags + special_tags:
        if tag not in dicom_tags:
            dicom_tags[tag] = None

    # add numerical tags to encoded tags dict directly as they are
    encoded_tags = {"enc_" + key: value for key, value in dicom_tags.items() if key in numerical_tags}

    # add feature indicating the number of unique values were found during aggregation of dicom tags
    # during aggregation, tags with multiple values were set to "n=<nr of unique values>"
    for tag in categorical_tags + numerical_tags + special_tags:
        if str(dicom_tags[tag]).startswith("n="):
            try:
                encoded_tags[f"enc_{tag}_multiple"] = int(str(dicom_tags[tag]).split("=")[1])
            except (IndexError, ValueError):
                log.warning(
                    f"Could not extract number of multiple values from aggregated DICOM tag {tag} with value {dicom_tags[tag]}. Setting {tag}_multiple to 0."
                )
                encoded_tags[f"enc_{tag}_multiple"] = 0
            if tag in numerical_tags:
                # encode numerical tags with multiple values as None
                encoded_tags[f"enc_{tag}"] = None
            if tag in categorical_tags + special_tags:
                # encode categorical tags/special tags with multiple values later
                dicom_tags[tag] = None
        else:
            encoded_tags[f"enc_{tag}_multiple"] = 0

    # add enc_ImagesInSeries feature
    if "ImagesInSeries" in dicom_tags and dicom_tags["ImagesInSeries"] is not None:
        encoded_tags["enc_ImagesInSeries"] = dicom_tags["ImagesInSeries"]
        encoded_tags["enc_ImagesInSeries_missing"] = 0
    else:
        encoded_tags["enc_ImagesInSeries"] = None
        encoded_tags["enc_ImagesInSeries_missing"] = 1

    # add binary features indicating missing values, also for special tags
    # (for categorical tags, missing values are encoded as different value)
    for tag in numerical_tags + special_tags:
        encoded_tags[f"enc_{tag}_missing"] = int(dicom_tags[tag] is None)

    # Special handling of PixelSpacing tag:
    # encode x and y pixel spacing as separate numerical features
    encoded_tags["enc_PixelSpacing_x"] = None
    encoded_tags["enc_PixelSpacing_y"] = None
    if (not encoded_tags["enc_PixelSpacing_missing"] > 0) and (not encoded_tags["enc_PixelSpacing_multiple"] > 0):
        # extract x and y pixel spacing values
        try:
            encoded_tags["enc_PixelSpacing_x"] = dicom_tags["PixelSpacing"][0]
            encoded_tags["enc_PixelSpacing_y"] = dicom_tags["PixelSpacing"][1]
        except (IndexError, TypeError):
            log.warning(
                f"PixelSpacing tag has unexpected format: {dicom_tags['PixelSpacing']} instead of a list of two floats. Encoding PixelSpacing_x and PixelSpacing_y as None."
            )
    # else already handled above

    # special handling of ContrastBolusAgent tag:
    if "ContrastBolusAgent" in dicom_tags and dicom_tags["ContrastBolusAgent"] is not None:
        # binary feature indicating if ContrastBolusAgent is empty string
        encoded_tags["enc_ContrastBolusAgent_empty"] = int(dicom_tags["ContrastBolusAgent"] == "")
    else:
        encoded_tags["enc_ContrastBolusAgent_empty"] = 2  # missing value

    # Special handling of AcquisitionPlane/ImageOrientationPatient tag:
    # If AcquisitionPlane is not present, but ImageOrientationPatient is present, compute it from ImageOrientationPatient,
    # this is usually the case for slice-wise dicom tags, where AcquisitionPlane was not created during aggregation
    # enc_ImageOrientationPatient_missing and enc_ImageOrientationPatient_multiple are already created above
    # Do not add: enc_AcquisitionPlane_multiple
    if "AcquisitionPlane" not in dicom_tags:
        if "ImageOrientationPatient" not in dicom_tags or dicom_tags["ImageOrientationPatient"] is None:
            dicom_tags["AcquisitionPlane"] = None
        else:
            dicom_tags["AcquisitionPlane"] = compute_orientation(dicom_tags["ImageOrientationPatient"])

    # Do not encode multiple ImageOrientationPatient as missing
    if encoded_tags["enc_ImageOrientationPatient_multiple"] > 0:
        encoded_tags["enc_ImageOrientationPatient_missing"] = 0

    # Define values for categorical dicom tags:
    categories = {
        "AngioFlag": ["Y", "N"],
        "AcquisitionPlane": ["AX", "SAG", "COR", "ORTHO", "ROT"],
        "ImageType": {
            "projection": ["PROJECTION", "MIP"],
            "se": ["SE", "M_SE"],
            "epi": ["EP", "EPI"],
            "derived": ["DERIVED", "TRACE"],
            "diffusion": ["DIFFUSION"],
            "adc": ["ADC"],
            "water": ["WATER", "W"],
            "fat": ["FAT", "F"],
            "inphase": ["IN_PHASE", "IP"],
            "oppphase": ["OPP_PHASE", "OUT_PHASE", "OP"],
            "asl": ["ASL"],
            "ir": ["IR"],
            "fe": ["FFE", "M_FFE"],
            "subtraction": ["SUB", "SUBTRACTION", "SUBTRACT"],
        },
        "Modality": {
            "MR": ["MR"],
            "CT": ["CT"],
            # Maybe later
            # "PT": ["PT"], # Pet
            # "US": ["US"], # Ultrasound
            # "NM": ["NM"], # Nuclear Medicine
            # "CR": ["CR"], # Computed Radiography
        },
        "MRAcquisitionType": ["1D", "2D", "3D"],
        "ScanningSequence": {
            "SE": ["SE"],
            "IR": ["IR"],
            "GR": ["GR"],
            "EP": ["EP"],
            "RM": ["RM"],
        },
        "ScanOptions": {
            "fatsat": ["FS", "SFS"],
        },
        "SeriesDescription": {
            "loc": ["loc", "scout", "survey"],
            "mrcp": ["mrcp"],
            "bolus": ["bolus"],
            "t1w": ["mprage", "mp-rage", "t1"],
            "asl": ["asl"],
            "flair": ["flair"],
            "b0b1": ["field"],
        },
        "SequenceVariant": {
            "SK": ["SK"],
            "MTC": ["MTC"],
            "SS": ["SS"],
            "TRSS": ["TRSS"],
            "SP": ["SP"],
            "MP": ["MP"],
            "OSP": ["OSP"],
        },
    }

    # encode categorical tags and add features to encoded_tags dict
    for category, values in categories.items():
        if isinstance(values, dict):
            new_encoded_tags = encode_nonfixed_category(dicom_tags[category], category, values)
            for tag_name in new_encoded_tags:
                encoded_tags[tag_name] = new_encoded_tags[tag_name]
        elif isinstance(values, list):  # values is a list of fixed possible values
            new_encoded_tags = encode_fixed_category(dicom_tags[category], category, values)
            for tag_name in new_encoded_tags:
                encoded_tags[tag_name] = new_encoded_tags[tag_name]
        # else: Can't happen, as categories are defined above

    # make sure all None values are converted to float (or np.nan) for numerical compatibility
    for key in encoded_tags:
        if encoded_tags[key] is None:
            encoded_tags[key] = np.nan
        try:
            float(encoded_tags[key])
        except (TypeError, ValueError):
            log.warning(f"Encoded DICOM tag {key} has non-numeric value {encoded_tags[key]}. Converting to np.nan.")
            encoded_tags[key] = np.nan

    # sort encoded tags by key name and convert to np.recarray
    encoded_tags_sorted = {key: encoded_tags[key] for key in sorted(encoded_tags.keys())}

    return encoded_tags_sorted


def generate_series_metadata_vector(
    slice_filenames: List[str], version: str = "version_1"
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Generate an encoded metadata vector for a DICOM series.

    Reads DICOM tags from all slice files, aggregates them across the series,
    and encodes the result into a flat numerical feature dictionary.

    Args:
        slice_filenames: List of file paths to the DICOM files belonging to
            the same series.
        version: Encoding scheme version to use (default ``"version_1"``).

    Returns:
        Tuple of:

        - ``encoded`` (``Dict[str, Any]``): Encoded metadata vector for the
          series (one dict with sorted keys).
        - ``raw_tags`` (``Dict[str, Dict[str, Any]]``): Raw DICOM tags for
          each slice, keyed by file path.
    """

    # 1. read selected tags from dicom headers and store them in a dict of dictionaries
    slice_dicom_tags = {}
    for slice_filename in slice_filenames:
        slice_tags = read_dicom_tags(slice_filename)
        slice_tags = check_dicom_tag_types(slice_tags, selected_dicom_tags=SELECTED_RAW_DICOM_TAGS)
        slice_dicom_tags[slice_filename] = slice_tags

    # 2. aggregate dicom tags for all slices belonging to the same DICOM series
    # check that all slice filenames are in the same folder and use folder name as series identifier
    series_filepath = os.path.split(slice_filenames[0])[0]
    if not all(os.path.split(slice_filename)[0] == series_filepath for slice_filename in slice_filenames):
        log.warning(
            f"Slice filenames are aggregated, but are not located in the same folder! Found different folders for series: {series_filepath} and {[os.path.split(slice_filename)[0] for slice_filename in slice_filenames if os.path.split(slice_filename)[0] != series_filepath]}. Proceeding with aggregation anyway."
        )
    # aggregate dicom tags into a single-row dataframe with index series_filepath and convert back to dict
    aggregated_dicom_tags = (
        aggregate_dicom_tags(pd.DataFrame(data=slice_dicom_tags.values()))
        .rename(index={0: series_filepath})
        .to_dict(orient="index")
    )

    # 3. generate vector (np.recarray)  with the encoded metadata that can be fed into an RF or neural network model
    encoded_metadata_recarray = encode_dicom_tags_by_version(aggregated_dicom_tags, encoding_version=version)[
        series_filepath
    ]

    # 4. return raw DICOM metadata, encoded vector derived from aggregated metadata
    return encoded_metadata_recarray, slice_dicom_tags


def generate_slicewise_metadata_vector(
    slice_filenames: List[str], version: str = "version_1"
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Generate per-slice encoded metadata vectors for a list of DICOM files.

    Each slice is encoded independently (no cross-slice aggregation).

    Args:
        slice_filenames: List of file paths to DICOM files.
        version: Encoding scheme version to use (default ``"version_1"``).

    Returns:
        Tuple of:

        - ``encoded`` (``List[Dict[str, Any]]``): Encoded metadata vector for
          each slice; order matches *slice_filenames*.
        - ``raw_tags`` (``Dict[str, Dict[str, Any]]``): Raw DICOM tags for
          each slice, keyed by file path.
    """

    # 1. read selected tags from dicom headers and store them in a dict of dictionaries
    slice_dicom_tags = {}
    for slice_filename in slice_filenames:
        slice_tags = read_dicom_tags(slice_filename)
        slice_tags = check_dicom_tag_types(slice_tags, selected_dicom_tags=SELECTED_RAW_DICOM_TAGS)
        slice_dicom_tags[slice_filename] = slice_tags

    # 2. generate vector ( np.recarray)  with the preprocessed metadata that can be fed into an RF or neural network model
    encoded_metadata_recarrays_dict = encode_dicom_tags_by_version(deepcopy(slice_dicom_tags), encoding_version=version)
    # reformat as list in the same order as input slice filenames
    encoded_metadata_recarrays_list = [
        encoded_metadata_recarrays_dict[slice_filename] for slice_filename in slice_filenames
    ]

    # 3. return raw DICOM metadata and encoded vector derived from raw metadata
    return encoded_metadata_recarrays_list, slice_dicom_tags
