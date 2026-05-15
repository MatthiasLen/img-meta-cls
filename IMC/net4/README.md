# `net4`

This directory currently contains shared helper code for Network v04 model construction.

It does not expose standalone training or inference scripts in this public repository snapshot. The active Duke experiment entry points for the multimodal Network v04 model live in:

- `IMC.net4_duke.train`
- `IMC.net4_duke.infer`
- `IMC.net4_duke.summarize_cv`

The central model-building utility here is `helper.py`, which is used by the Duke training and inference pipeline to keep architecture construction consistent across training and checkpoint loading.
