"""IMC Network v06 – PixelOnlyModel (image-only) on Duke Liver Dataset.

This sub-package contains experiment scripts for training and evaluating
:class:`~IMC.network06.PixelOnlyModel` on the Duke Liver MRI dataset with
an optional Random-Forest metadata gate for ``SequenceType_Code_norm``.

Entry points
------------
- :mod:`IMC.net6.train`       – 5-fold cross-validation training (image model).
- :mod:`IMC.net6.infer`       – Batch inference with heuristic RF gate.
- :mod:`IMC.net6.train_rf`    – Train per-fold Random Forest classifiers on
                                Duke tabular metadata.
"""
