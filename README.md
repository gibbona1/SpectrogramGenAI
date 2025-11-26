<!-- badges: start -->
[![arXiv](https://img.shields.io/badge/arXiv-2412.01530-b31b1b.svg)](https://arxiv.org/abs/2412.01530)
<!-- badges: end -->
# Generative AI-based data augmentation for improved bioacoustic classification in noisy environments

## Overview

This repository implements generative AI-based data augmentation techniques for improving bioacoustic classification in noisy environments, particularly for bird species detection at wind farm sites. We explore Auxiliary Classifier Generative Adversarial Networks (ACGAN) and Denoising Diffusion Probabilistic Models (DDPMs) to synthesize spectrograms, enhancing training data diversity without requiring extensive expert labeling.

## Dataset

The project includes a new audio dataset of 640 hours of bird calls recorded at wind farm sites in Ireland. Approximately 800 samples are expert-labeled, providing a challenging benchmark due to background wind and turbine noise.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/gibbona1/SpectrogramGenAI.git
    cd SpectrogramGenAI
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    Ensure you have Python 3.8+.

## Usage

### Data Augmentation with Generative Models

1. **Prepare Data**: Either `.wav` files or 256x256 spectrograms

2. **Train ACGAN**:
    ```bash
    python train_acgan.py
    ```

3. **Train DDPM**:
    ```bash
    python train_ddpm.py
    ```

4. **Generate Synthetic Spectrograms**:
    ```bash
    python generate_spectrograms.py --model ddpm --num_samples 1000 --output_path data/synthetic/
    ```

### Quality Metrics

Inception Score
```bash
python inception_score.py <image_folder>
```

Fréchet Inception Distance
```bash
pip install pytorch-fid
python -m pytorch_fid folder1 folder2
```

Fréchet Audio Distance
```bash
python frechet_audio_distance.py --bg_dir folder1 --eval_dir folder2
```

### Classification Training

Train all classifiers on combined real and synthetic data:
```bash
python train_classifiers.py
```

For more details, refer to the [arXiv preprint](https://arxiv.org/abs/2412.01530).