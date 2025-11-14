
# Brain EIT: Executive Summary

Electrical Impedance Tomography (EIT) uses tiny, safe electrical currents and boundary voltage measurements to infer conductivity changes inside the head without radiation. In this project we build brain EIT difference images: from pairs of baseline vs changed measurements we predict where conductivity has shifted (e.g., potential anomaly regions). A streamlined dataset (complex voltage differences + 128×128 conductivity change maps) feeds an Attention U‑Net that learns to turn noisy boundary data into clearer interior images. Training logs show a large reduction in validation error, and saved checkpoints let results be reproduced. Figures below show the model structure and typical dataset characteristics.

# Brain EIT: What this project does and why it matters

This project explores a gentle, radiation‑free way to “see” inside the head called Electrical Impedance Tomography (EIT). Instead of taking a photo like an X‑ray or CT, EIT sends tiny, safe electrical currents through the scalp using electrodes placed around the head. By measuring how those currents travel, we can estimate what’s happening inside the brain and reconstruct an image that highlights differences in tissue or fluid.

Our work uses modern AI to make these EIT images clearer and more reliable, so clinicians and researchers can get faster insights with minimal risk.


## EIT in simple terms

- The body conducts electricity differently depending on what’s inside (blood, brain tissue, cerebrospinal fluid, etc.).
- We place a ring of small electrodes on the head. A very small current is injected between some electrodes; others measure the resulting voltages.
- From those measurements, we compute a map of how well different regions conduct electricity — this is the EIT image.

Why this is promising for the brain:
- It’s non‑invasive and does not use ionizing radiation.
- It can be made portable and relatively affordable.
- It could offer rapid monitoring at the bedside, for example to detect bleeding, swelling, or other changes.


## The challenge we address

Turning surface voltage measurements into a meaningful image is a very hard math problem. Small measurement errors can cause big changes in the reconstruction, making images blurry or noisy. Traditional methods can struggle, especially when the signal is weak or the head shape varies.

We tackle this by training AI models to recognize patterns in the measurements and produce cleaner, more accurate EIT images.


## What we incorporated (at a glance)

- Curated brain EIT datasets stored in efficient HDF5 files for training, validation, and testing.
- Two AI model families that are widely used in medical imaging:
  - U‑Net: extracts multi‑scale features to form a clean image.
  - Attention U‑Net: adds a “focus” mechanism so the model pays more attention to important regions.
- Saved best models so results can be reproduced without re‑training.
- Training logs that show how the models improved over time.


## How the datasets are generated (conceptually)

When building EIT datasets, we follow a standard and safe simulation-and-measurement approach:

1) Define the head setup
- Place a ring of electrodes around a head model (or mannequin/phantom in lab tests).
- Specify how current is injected (which pair of electrodes) and which electrodes measure the response.

2) Establish a baseline and a change
- Baseline: normal brain conductivity map.
- Change: introduce a localized anomaly (e.g., bleed, edema) by slightly altering conductivity in a region.

# Brain Electrical Impedance Tomography (EIT) Reconstruction with Attention U‑Net

This repository studies brain EIT image reconstruction with a learned, end‑to‑end approach based on an Attention U‑Net. It documents the forward physics, the inverse problem, our dataset design, and the reconstruction architecture used to map complex boundary voltage differences to conductivity difference images.


## 1. Background: EIT forward physics and measurements

Let $\Omega$ denote the head domain with conductivity $\sigma(\mathbf{x})>0$. Under steady current injection, the electric potential $u(\mathbf{x})$ satisfies the conductivity equation inside the domain:

$$\nabla\cdot(\sigma\,\nabla u)=0 \quad \text{in } \Omega.$$

At the scalp, we place electrodes. In practice we:
- drive a known current between a pair of electrodes (the net injected current over all electrodes sums to zero),
- measure the resulting voltages at the electrodes,
- assume the rest of the boundary is effectively insulating.

These measurements are complex‑valued (magnitude and phase). By repeating multiple current patterns, we collect a vector of measurements per sample.

Difference imaging. Rather than recover absolute $\sigma$, we form differences relative to a baseline. The reconstruction task is to estimate a conductivity change image $\Delta\sigma$ that explains $\Delta \mathbf{V}$ under the physics above. We learn this mapping directly with a neural network.


## 2. Dataset design and structure

Data units. Each sample consists of:

- Complex voltage differences $\delta\mathbf{V} \in \mathbb{C}^M$ gathered across a set of electrode injection/measurement configurations (stacked real and imaginary parts).
- A ground‑truth 2D conductivity difference image $\delta\sigma \in \mathbb{R}^{H\times W}$ (e.g., $128\times128$ grid).

Generating examples (concept):
1) Choose a baseline head model and electrode layout. 2) Introduce an anomaly (e.g., focal bleed/edema) by adjusting conductivity within a subregion. 3) Solve the forward problem to obtain voltages for baseline and perturbed states; take the difference. 4) Optionally add realistic noise $\boldsymbol{\eta}$ to both real and imaginary parts, e.g., $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 \mathbf{I})$. 5) Store the resulting pair $(\delta\mathbf{V}, \delta\sigma)$.

On‑disk structure (HDF5):
- `difference`: the complex voltage‑difference vector for each sample (length may vary by protocol),
- `gt_difference`: the target difference image for that sample ($H\times W$ array).


## 3. Reconstruction model (Attention U‑Net)

Goal. Learn to convert the measured voltage differences into an image showing where conductivity changed.

How the model is organized (plain language):
- Step 1: Flattened measurements are first “shaped” into a 2D image canvas (128×128) using a simple learnable projection. This gives the network a starting picture to work with.
- Step 2: The U‑Net encoder reads the image at multiple scales (from coarse to fine), learning patterns that relate to real changes in tissue conductivity.
- Step 3: The decoder rebuilds the image back to full size. At each step it brings back details from earlier layers using skip connections.
- Step 4: Attention blocks act like a spotlight on those skip connections, highlighting the parts that matter most and dimming the rest (noise and irrelevant regions).
- Step 5: A final light-weight layer produces the predicted conductivity‑change image.

Training signal (what the model tries to get right):
- Match the predicted image to the ground truth (mean‑squared error).
- Keep edges and boundaries in the right place (a small gradient‑based term).
- Preserve overall structure (a small SSIM‑based term).

Putting it together: the model mainly minimizes pixel‑wise error, with two gentle nudges to keep shapes and edges realistic. We also use standard training safeguards (learning‑rate scheduling, early stopping, best‑model checkpoint).

Figure: Attention U‑Net schematic used in this work:

![Attention U‑Net Architecture](unet_attention_architecture.svg)

<sub>Figure 1: The measurement vector is projected to a 2D canvas, processed through an encoder (blue) and decoder (green) with attention gates (red) highlighting useful skip features; bottleneck (orange) provides deep context. SVG for clarity; PNG fallback at `docs/figures/unet_attention_architecture.png`.</sub>


## 4. Reconstruction pipeline (end‑to‑end)

1) Load $(\delta\mathbf{V}, \delta\sigma)$ pairs from HDF5 (`difference`, `gt_difference`).
2) Build input vectors by stacking real/imag parts; pad to $2M_\max$.
3) Standardize inputs and targets (dataset mean/std).
4) Map inputs to a $128\times128$ image via a dense layer; pass through Attention U‑Net.
5) Train with the composite loss and callbacks; monitor validation metrics and save the best model.
6) At inference, apply the same pre‑processing and predict $\hat{\delta\sigma}$; optionally de‑standardize to original scale.


## 5. Results summary (training log)

From `training_log.csv` (100 epochs):
- Validation loss improved from ~2.47 at epoch 0 to ~0.20 by the end.
- Best observed validation loss ≈ 0.1987 (around epoch 90).
- Final epoch (99): val\_loss ≈ 0.2085, val\_MAE ≈ 0.1044, val\_MSE ≈ 0.1993.

These trends indicate the model learns a stable mapping that generalizes across held‑out data under the given normalization and padding scheme.


## 6. Dataset visualization

Below is a high-level view of the dataset characteristics used in this work (inputs and targets). This helps contextualize the magnitude and distribution of measurements and ground-truth images across the splits.

![Dataset visualization](data_visualization.png)

<sub>Figure 2: Example dataset summary illustrating typical input measurement distributions and corresponding conductivity change targets (visual layout depends on the generated plot). Helps confirm dynamic range and anomaly localization diversity.</sub>


## 7. Repository layout (key artifacts only)

- Notebooks
  - `FUSE_UNET.ipynb` — model and training/evaluation pipeline used in this work
  - `BRAIN_EIT.ipynb` — EIT context, data preparation, and analysis notebook
- Models
  - `best_attention_unet_eit.keras` — best checkpoint (by validation loss)
  - `eit_attention_unet_final.keras` — final saved model after training
- Datasets & meshes
  - `brain_eit_training*/` `brain_eit_validation*/` `brain_eit_test*/` — contain `brain_eit_dataset.h5`
  - `mesh_POOL.pkl` — mesh/forward model pool used during data generation (if provided in your setup)
- Logs & metrics
  - `logs/train/` and `logs/validation/` — TensorBoard event files
  - `training_log.csv` — per‑epoch metrics (loss, MAE, MSE)
- Results & visuals
  - `sample_prediction_0.png` — example qualitative result (saved after running inference)
  - `data_visualization.png` — overview plot of dataset distributions/structure
- Documentation
  - `README.md` — this document


## 7. Notes on noise and robustness

- Measurement noise enters both real and imaginary components. During simulation or preprocessing, additive zero‑mean Gaussian noise can approximate contact and electronics effects; SNR can be tuned to stress‑test robustness.
- Variable measurement counts (due to protocol changes or dropped frames) are addressed by padding to the dataset maximum and learning an embedding that is resilient to trailing zeros.
- Attention gates improve tolerance to spurious features by weighting skip pathways based on decoder context.


## 8. References (selected)

- B. Holder (ed.), Electrical Impedance Tomography: Methods, History and Applications, IOP Publishing, 2004.
- A. Adler and W. R. B. Lionheart, “Uses and abuses of EIT,” Physiological Measurement, 2006.
- O. Ronneberger, P. Fischer, T. Brox, “U‑Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI 2015.
- O. Oktay et al., “Attention U‑Net: Learning Where to Look for the Pancreas,” arXiv:1804.03999, 2018.
- Z. Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE TIP, 2004.
- PyEIT: https://github.com/liubenyuan/pyEIT



