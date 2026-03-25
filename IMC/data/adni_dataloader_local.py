"""
Backward-compatibility shim.

``adni_dataloader_local`` was generalised into ``brain_dataloader_local``.
All symbols are re-exported so existing imports continue to work unchanged.

New code should import from :mod:`IMC.data.brain_dataloader_local` directly.
"""
from IMC.data.brain_dataloader_local import (  # noqa: F401
    BrainDataset as ADNIDataset,
    Brain3DDataset as ADNI3DDataset,
    get_train_dataloader,
    get_valid_dataloader,
    get_test_dataloader,
    get_infer_dataloader,
)
from IMC.data.constants import ADNI_LABEL_NAMES  # noqa: F401

__all__ = [
    "ADNIDataset",
    "ADNI3DDataset",
    "ADNI_LABEL_NAMES",
    "get_train_dataloader",
    "get_valid_dataloader",
    "get_test_dataloader",
    "get_infer_dataloader",
]
