"""IMC Network v07 – PyramidPooling3DClassifier on Duke and ADNI datasets.

This sub-package contains experiment scripts for training and evaluating
:class:`~IMC.network07.PyramidPooling3DClassifier` on two volumetric MRI
datasets.  Each dataset has its own dedicated training and inference entry
points; both share the same underlying architecture.

Entry points – Duke Liver MRI
------------------------------
- :mod:`IMC.net7.train_duke`   – Single-fold CV training on the Duke dataset
                                  (run once per fold with ``--fold``).
- :mod:`IMC.net7.infer_duke`   – Batch inference on the Duke dataset.

Entry points – ADNI Brain MRI
------------------------------
- :mod:`IMC.net7.train_adni`   – Single-fold CV training on the ADNI dataset
                                  (run once per fold with ``--fold``).
- :mod:`IMC.net7.infer_adni`   – Batch inference on the ADNI dataset.
"""
