"""
Constants for IMC data module.

This module contains label definitions that are used across multiple modules.
These constants don't have any dependencies on torch or other heavy libraries,
making them safe to import in lightweight inference environments.
"""

DUKE_ORIGINAL_LABEL_NAMES = {
    "SequenceType_Code_norm": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
}

# Human-readable names for Duke sequence type letter codes, used in CV summaries
LABEL_NAME_MAPS = {
    "Arterial T1w": ["C", "O", "Q"],
    "Portven T1w": ["K"],
    "Late T1w": ["E", "N", "P"],
    "AX T2w": ["A"],
    "COR T2w": ["J"],
    "AX FatSat T1w": ["B"],
    "AX Dixon In": ["G"],
    "AX Dixon Opp": ["H"],
    "AX DWI": ["I"],
    "AX ADC": ["M"],
    "Localizer": ["L"],
    "MRCP": ["D"],
    "Other": ["F"],
}

# Reverse mapping: letter code -> label name
LETTER_TO_LABEL_NAME: dict[str, str] = {}
for _label_name, _letter_codes in LABEL_NAME_MAPS.items():
    for _letter in _letter_codes:
        LETTER_TO_LABEL_NAME[_letter] = _label_name

# Sort order: letter code -> index (preserves LABEL_NAME_MAPS order)
LETTER_SORT_ORDER: dict[str, int] = {}
_order_idx = 0
for _label_name, _letter_codes in LABEL_NAME_MAPS.items():
    for _letter in _letter_codes:
        LETTER_SORT_ORDER[_letter] = _order_idx
        _order_idx += 1
