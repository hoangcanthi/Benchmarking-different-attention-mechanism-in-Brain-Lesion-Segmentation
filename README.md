## Viet Hoang Pham - City, University of London
## Project title: Benchmarking Different Attention Mechanism in Brain Lesion Segmentation - Supervised by: Dr Atif Riaz

This repo contains training, evaluation, and visualization utilities built around nnU‑Net v2 for 2D stroke lesion segmentation (ATLAS2).

Reference: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

### Setup
- Create a virtual environment and install required libraries in requirements.txt
- Set nnU‑Net path variables in your shell before training/prediction, for guidance please refer to nnUNet official rep here: https://github.com/MIC-DKFZ/nnUNet

```bash
# PowerShell (edit paths)
$env:nnUNet_raw= #your path to raw organised dataset folder
$env:nnUNet_preprocessed= # your path to preprocessed folder
$env:nnUNet_results= # your path to results folder
```

All training outputs and summaries are expected under `nnUNet_results` as a convention [[training files in nnUNet_results]].

### Data preprocessing (ATLAS2): run preprocess.py, then CLI: nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity (in our case we use 201 for datasetID)

### Training
- Plain 2D nnU‑Net v2:
```bash
nnUNetv2_train 201 2d 0 -tr nnUNetTrainer__nnUNetPlans__2d
```
- Custom attention trainers: must be placed in nnunetv2/training/custom to be recognised by nnunet, below are the CLIs to run the training
```bash
# CBAM 
nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_CBAM

# Criss‑Cross (axial) attention
nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_CrissCross

# Squeeze‑and‑Excitation (SE)
nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_SE

# Attention Gates (AG)
nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_AttentionGate
```

Notes
- Folds: repeat `-f 0..4` for cross‑validation.
- Models are stored under `nnUNet_results/Dataset201_ATLAS2/<trainer>/fold_<k>/`.

### Inference
Predict on test images (replace trainer to match trained model):
```bash
nnUNetv2_predict \
  -d 201 -c 2d -f 0 \
  -tr nnUNetTrainer__nnUNetPlans__2d \
  -i nnUNet_raw/Dataset201_ATLAS2/imagesTs \
  -o predictions_plain
```

### Evaluation
- HD95 from predictions vs labels: run extract_hd95_from_summaries.py (provide summ)


### Visualization
- Overlays of predictions on images: in Custom scipts: overlay_fold3_grid.py to produce validation prediction overlay


### Notes on architecture and configs (2D PlainConvUNet) - this is typically done by nnunet itself but worth mentioning
- Patch size: typically 256×224.
- Encoder/decoder: six stages, two 3×3 convs per stage; InstanceNorm(ε=1e‑5, affine=True) + LeakyReLU(0.01).
- Downsampling by 3×3 stride‑2 convs in encoder; upsampling by 2×2 stride‑2 transposed convs in decoder.
- Final head: 1×1 conv to 2 logits (background/lesion); Softmax applied at inference.

### Reproducibility tips
- Ensure environment variables are set in the shell where you run `nnUNetv2_*` commands.
- Use consistent dataset IDs (201) and folder names: `nnUNet_raw/Dataset201_ATLAS2`.
- Keep predictions and figures under `nnUNet_results` for organization.


