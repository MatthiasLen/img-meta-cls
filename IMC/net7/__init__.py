"""IMC Network v07 – PyramidPooling3DClassifier on the Duke Liver MRI dataset.

This sub-package contains experiment scripts for training and evaluating
:class:`~IMC.network07.PyramidPooling3DClassifier` on the Duke Liver MRI
dataset.

Entry points – Duke Liver MRI
------------------------------
- :mod:`IMC.net7.train_duke`   – Single-fold CV training on the Duke dataset
                                  (run once per fold with ``--fold``).
- :mod:`IMC.net7.infer_duke`   – Batch inference on the Duke dataset.
"""
