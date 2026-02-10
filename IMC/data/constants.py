"""
Constants for IMC data module.

This module contains label definitions and feature selections that are used
across multiple modules. These constants don't have any dependencies on
torch or other heavy libraries, making them safe to import in lightweight
inference environments.
"""

# Default label mappings for medical imaging classification
DEFAULT_LABEL_NAMES = {
    "label_SequenceType": [
        "T1", "T2", "DWI", "ADC", "SUB", "DIXON_F", 
        "DIXON_IN", "DIXON_OPP", "BOLUS", "OTHER", "na"
    ],
    "label_FatSat": ["yes", "no", "na"],
    "label_MRCP": ["yes", "no", "na"],
    "label_AcquisitionPlane": ["AX", "COR", "SAG", "ORTHO", "ROT", "na"],
    "label_ContrastPhase": ["pre", "art", "portven", "trans", "hepa", "na"],
    "label_Contrast": ["pre", "post", "na"],
    "label_Localizer": ["yes", "no", "na"],
}

DUKE_LABEL_NAMES = {
    "label_SequenceType": ["T1", "T2", "DWI", "ADC", "DIXON_IN", "DIXON_OPP", "OTHER", "na"],
    "label_FatSat": ["yes", "no", "na"],    
    "label_MRCP": ["yes", "no", "na"],
    "label_AcquisitionPlane": ["AX", "COR", "OTHER", "na"],
    "label_ContrastPhase": ["pre", "art", "portven", "late", "na"],
    "label_Contrast": ["pre", "post", "na"],
    "label_Localizer": ["yes", "no", "na"],
}

# Selected DICOM metadata features for model inference
# These features are extracted from encoded DICOM tags and used as input
# to the neural network model alongside image data
SELECTED_FEATURES = [
    'enc_AcquisitionPlane_ORTHO',
    'enc_SeriesDescription_mrcp',
    'enc_SeriesDescription_loc',
    'enc_FlipAngle',
    'enc_ScanningSequence_SE',
    'enc_AcquisitionPlane_COR',
    'enc_MRAcquisitionType_3D',
    'enc_MRAcquisitionType_2D',
    'enc_ScanOptions_fatsat',
    'enc_AcquisitionPlane_AX',
    'enc_AcquisitionPlane_ROT',
    'enc_ImageType_adc',
    'enc_SliceLocation_multiple',
    'enc_SliceThickness',
    'enc_ImageType_subtraction',
    'enc_EchoTrainLength',
    'enc_ImageType_water',
    'enc_DiffusionBValue_missing',
    'enc_ScanningSequence_EP',
    'enc_SequenceVariant_SK',
    'enc_PixelBandwidth',
    'enc_ImageOrientationPatient_multiple',
    'enc_ContrastBolusAgent_missing',
    'enc_SeriesDescription_t1w',
    'enc_SequenceVariant_SS',
    'enc_AcquisitionPlane_SAG',
    'enc_AcquisitionDuration_missing',
    'enc_PixelSpacing_y',
    'enc_NumberOfAverages',
    'enc_EchoTime',
    'enc_ImagesInSeries',
    'enc_PixelSpacing_x',
]
