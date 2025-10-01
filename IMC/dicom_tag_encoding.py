from ast import literal_eval
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("dataloader")

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

    # drop all other columns:
    all_columns = ["Filepath"] + categorical_tags + numerical_tags + added_features

    tags_to_drop = [col for col in encoded_dicom_tags_df.columns if col not in all_columns]
    encoded_dicom_tags_df = encoded_dicom_tags_df.drop(columns=tags_to_drop, errors="ignore")

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

    encoded_dicom_tags_df = encoded_dicom_tags_df.reindex(columns=sorted(encoded_dicom_tags_df.columns))

    # make sure all values are numeric now:
    for col in encoded_dicom_tags_df.columns:
        if col != "Filepath":  # keep Filepath as string
            # convert to numeric, coerce errors to NaN
            encoded_dicom_tags_df[col] = pd.to_numeric(encoded_dicom_tags_df[col], errors="coerce")

    return encoded_dicom_tags_df
