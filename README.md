# SDF Extension for Weakly Supervised 4D Cardiac Reconstruction

This repository contains the **SDF-specific adaptation** developed for a final year project on weakly supervised reconstruction of four-dimensional (4D) cardiac geometry from echocardiography.

The project investigates whether **signed distance functions (SDFs)** can act as an alternative implicit geometric representation to meshes within an existing weakly supervised latent-translation framework.

## Acknowledgement

This project builds on and adapts ideas and code structure from the original
[4DHeartModel](https://github.com/laumerf/4DHeartModel) repository by Fabian Laumer and co-authors, which accompanies the paper *Weakly supervised inference of personalized heart meshes based on echocardiography videos*.

```bibtex
@article{laumer2022weakly,
  title={Weakly supervised inference of personalized heart meshes based on echocardiography videos},
  author={Laumer, Fabian and Amrani, Mounir and Manduchi, Laura and Beuret, Ami and Rubi, Lena and Dubatovka, Alina and Matter, Christian M and Buhmann, Joachim M},
  journal={Medical Image Analysis},
  pages={102653},
  year={2022},
  publisher={Elsevier}
}
```

## Repository Scope

This repository is intentionally focused on the **new SDF contribution**, not the full inherited mesh/echo codebase.

Included here are:

- mesh-video to SDF preprocessing and global tensor construction
- SDF video autoencoder training code
- SDF latent ejection fraction (EF) prediction code
- SDF latent caching helpers
- SDF-compatible CycleGAN implementation
- SDF-specific experiment configurations
- selected documentation and architecture figures

Not included here are:

- raw datasets
- generated SDF datasets
- experiment outputs and trained weights
- most of the inherited mesh and echocardiography pipeline

## Repository Structure

- `sdf_pipeline/preprocessing/`
  - mesh-video to SDF conversion helpers and global tensor construction scripts
- `sdf_pipeline/autoencoder/`
  - SDF video autoencoder model, data loader, and training script
- `sdf_pipeline/ef_predictor/`
  - EF prediction from frozen SDF latents
- `sdf_pipeline/cyclegan/`
  - SDF-compatible latent CycleGAN code and latent-cache helpers
- `sdf_pipeline/data/`
  - SDF dataset utilities used by the integrated pipeline
- `sdf_pipeline/configs/`
  - SDF experiment configuration files
- `docs/`
  - supporting diagrams and notes

## Pipeline Overview

The overall SDF branch follows this flow:

```text
mesh video NPZ
  -> preprocessed to SDF video
  -> global multichannel SDF tensor
  -> SDF autoencoder latent
  -> SDF EF predictor / SDF-compatible latent CycleGAN
```

- **Mesh-video to SDF preprocessing**
  - `batch_convert_meshvideo_to_sdf.py` converts mesh-video `.npz` files into intermediate component-wise `*_sdf_video.npz` files.
  - `preprocess_sdf_data_to_global_tensors.py` converts those SDF videos into globally aligned `*_global_sdf.npz` tensors for model training.

- **SDF autoencoder**
  - The autoencoder learns to reconstruct global multichannel SDF videos from a compact video-level latent vector.
  - This latent vector is the SDF-domain representation used by the downstream EF predictor and CycleGAN helpers.

- **SDF EF predictor**
  - The EF predictor freezes the trained SDF autoencoder, encodes SDF videos into latents, and trains a small regressor to predict ejection fraction labels such as `EF_Biplane` or `EF_Vol`.
  - This provides an SDF-latent analogue of the EF supervision used in the original mesh-based pipeline.

- **SDF-compatible CycleGAN**
  - The CycleGAN adapts the original latent translation setup from `echo latent <-> mesh latent` to `echo latent <-> SDF latent`.
  - In this version, generated SDF latents can be decoded by the frozen SDF autoencoder and supervised with the frozen SDF EF predictor.

## Environment

Two environment descriptions are provided:

- `requirements.txt`
- `environment.yml`

These are included for reproducibility, but they may still require adjustment depending on the target machine and any external dependencies inherited from the original framework.
