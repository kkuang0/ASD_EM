# ASD_EM

This repository contains utilities for electron microscopy (EM) analysis.  The
`denoising` module implements a self‑supervised Noise2Void pipeline for gentle
denoising of axon cross‑section images.

## Noise2Void Usage

### Training

```
python -m src.denoising.train_n2v \
    --images /path/to/*.tif \
    --output-dir n2v_runs \
    --patch-size 64
```
Example denoised patches are saved to the output directory after each epoch for visual inspection.

### Inference

```
python -m src.denoising.inference_n2v \
    --image sample.tif \
    --model n2v_runs/best.pt \
    --output sample_denoised.tif
```

Validation utilities such as edge preservation and texture similarity are
available in `src/denoising/validation.py`.
