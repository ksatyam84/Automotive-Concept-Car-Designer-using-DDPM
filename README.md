# 🚗 Automotive Concept Car Designer using DDPM

## Overview

This project implements a Denoising Diffusion Probabilistic Model (DDPM) to generate novel automotive concept car designs. Leveraging an attention-based U-Net architecture and the Stanford Cars dataset, the model learns to synthesize realistic and unique car images from random noise. The project demonstrates advanced generative AI for industrial design, including hybrid car generation via latent space interpolation and quantitative/qualitative evaluation of results.

## Key Features

- **Attention U-Net Architecture**: Utilizes Hugging Face Diffusers' UNet2DModel with self-attention for global geometric consistency.
- **DDPM Training**: Trains on 16,185 images (196 classes) from the Stanford Cars dataset, downsampled to 64x64 resolution.
- **Latent Space Interpolation**: Generates hybrid car concepts by interpolating between noise vectors.
- **Novelty Check**: Nearest neighbor analysis ensures generated designs are not memorized from training data.
- **Evaluation Metrics**: Uses Fréchet Inception Distance (FID) to assess image quality and realism.
- **Exponential Moving Average (EMA)**: Improves sample quality during generation.

## Results

- **FID Score**: Achieved FID < 50, indicating high-quality, realistic generations.
- **Hybrid Car Generation**: Demonstrated smooth transitions and meaningful representations in latent space.
- **Novelty**: Nearest neighbor analysis confirms the model creates new designs, not memorized copies.

## Usage

1. **Setup**: Install dependencies (PyTorch, torchvision, diffusers, accelerate, kagglehub).
2. **Data**: Download the Stanford Cars dataset from Kaggle.
3. **Training**: Run the notebook to train the DDPM model. Checkpoints and sample images are saved automatically.
4. **Generation**: Use the trained model to generate new car concepts and hybrid designs.
5. **Evaluation**: Compute FID score and perform novelty checks using provided notebook cells.

## File Structure

- `final_project.ipynb`: Main notebook with code, documentation, and results.
- `README.md`: Project summary and instructions.
- `checkpoints/`: Saved model checkpoints.
- `samples/`: Generated images and visualizations.

## Technical Highlights

- **Model**: UNet2DModel (Diffusers), 4 blocks, attention at bottleneck, sinusoidal time embeddings.
- **Training**: MSE loss, AdamW optimizer, cosine LR schedule with warmup, EMA for stable sampling.
- **Evaluation**: FID score, nearest neighbor analysis for novelty.

## Future Work

- Train on higher resolutions (128x128, 256x256)
- Add class-conditional generation (SUV, sedan, etc.)
- Explore latent diffusion and classifier-free guidance

## Applications

- AI-assisted automotive design brainstorming
- Rapid concept prototyping
- Design space exploration
- Hybrid style generation

---

**This project bridges deep learning research with creative industrial design, showcasing the power of generative AI for real-world applications.**
