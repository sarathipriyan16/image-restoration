# KLA Hackathon Submission — Image Restoration

This project implements a custom image-restoration model for restoring noisy grayscale semiconductor images. The solution is built around a lightweight edge-aware ConvNeXt-V2 style restoration network and is designed to process `.npy` image inputs and produce restored `.npy` outputs.

## Project Goal

The model is designed to denoise and recover clean image content from degraded inputs using:

- Edge-aware feature extraction
- Residual ConvNeXt-V2 blocks
- Dataset-statistics-based normalization
- A combined restoration loss balancing pixel fidelity and edge preservation

## Repository Structure

```text
image-restoration/
├── run.py
├── requirements.txt
├── README.md
└── models/
    └── trial_18_model.pth
```

## Model Design

The submitted model uses a residual encoder-decoder restoration architecture with the following components:

- `LaplacianEdgeExtractor` — computes edge maps from the degraded input
- `ConvNeXtV2Block` — depthwise convolution, Layer Normalization, GELU activation, and Global Response Normalization
- `RepPhyDAS_ConvNeXt` — encoder-decoder restoration network with residual connections and skip connections
- `CombinedRestorationLoss` — combines Charbonnier loss with an edge-aware loss during training

The edge-aware design helps preserve fine structural details while restoring degraded image content.

## Data Normalization

During training, dataset statistics were used to normalize the input and target arrays.

The final trained checkpoint stores the normalization parameters (`a` and `b`) together with the model weights. Therefore, `run.py` does not require `dataset_statistics.json` or any additional dataset files during inference.

## Training

The model was trained using:

- AdamW optimizer
- Cosine learning-rate decay
- Gradient clipping
- Mixed-precision training on CUDA-enabled systems
- Combined Charbonnier and edge-aware restoration loss

The final trained checkpoint is:

```
models/trial_18_model.pth
```

## Requirements

Install the required dependencies using:

```
pip install -r requirements.txt
```

## Running the Model

The solution accepts a directory containing `.npy` input images and writes the restored `.npy` files to the specified output directory.

Run:

```
python run.py <input-dir> <output-dir>
```

Example:

```
python run.py ./test_data ./outputs
```

The output directory is created automatically if it does not exist.

Each input `.npy` file produces exactly one output `.npy` file with the same filename.

## Output Format

The generated outputs satisfy the following format:

- NumPy `.npy` files
- Grayscale images
- Output shape: `(H, W)` or `(H, W, 1)`
- Data type: `float32`
- Pixel values in the range `[0, 1]`
- No `NaN` or `Inf` values
- Same filename as the corresponding input

## Hardware and Offline Execution

The inference script automatically uses an NVIDIA GPU when CUDA is available:

```
CUDA available → NVIDIA GPU
CUDA unavailable → CPU fallback
```

The model weights are included in the repository. Inference does not require:

- Internet access
- API keys
- Additional model downloads
- User interaction
- Manual configuration
- External datasets

## Results

The final inference pipeline was tested on 400 `.npy` test images.

The complete test run successfully generated 400 corresponding `.npy` outputs.

All 400 generated outputs were automatically validated for:

- Correct output count
- Matching filenames
- Valid grayscale dimensions
- Values within `[0, 1]`
- Absence of `NaN` and `Inf` values

The validation result was:

```
Input files: 400
Output files: 400
Missing outputs: 0
Unexpected outputs: 0
Invalid outputs: 0
STATUS: PASS
```

## Summary

This repository provides a complete offline inference solution for degraded grayscale image restoration using an edge-aware ConvNeXt-V2 style architecture. The submission includes the inference script, trained model weights, dependency specification, and execution instructions required to reproduce the restoration pipeline.
