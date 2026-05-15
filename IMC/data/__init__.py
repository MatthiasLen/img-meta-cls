"""Data loading and metadata encoding modules for IMC."""

from importlib import import_module

__all__ = [
    "augment",
    "constants",
    "dicom_tag_encoding_v2",
    "duke_dataloader_3d",
    "duke_dataloader_local",
    "encode_metadata",
    "image_reader",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
