# VGG-Splat using Depth Regularization

## 1. Introduction

This project is a **course project from students of Sun Yat-sen University (SYSU)**. We have built a comprehensive pipeline for **3D scene reconstruction and novel view synthesis** by integrating several state-of-the-art open-source works.

We would like to express our gratitude to the authors of the following projects for their excellent contributions:

- **VGGT**: Visual Geometry Grounded Transformer for robust camera pose estimation.  
  https://github.com/facebookresearch/vggt

- **3D Gaussian Splatting**: Real-time radiance field rendering.  
  https://github.com/graphdeco-inria/gaussian-splatting

- **Depth Anything V2**: High-quality monocular depth estimation.  
  https://github.com/DepthAnything/Depth-Anything-V2

---



## 2. Getting Started

To run the code, please follow the steps below.

### Step 1: Environment Setup

```bash
# Install basic dependencies
pip install -r requirements.txt

# Build and install Gaussian Splatting submodules
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
cd ..
```

### Step 2: Prepare Checkpoints

You need to manually download and place the model checkpoints into the designated directories:

* **VGGT**
  Download the required checkpoints and place them in `vggt/checkpoints/`:

  ```text
  vggt/checkpoints/
    ├── vggt.pt                    # Main VGGT model
    ├── vggsfm_v2_tracker.pt                # VGGSfM tracker
    ├── dinov2_vitb14_reg4_pretrain.pth     # DINOv2 backbone
    └── aliked-n16.pth                      # ALIKED feature extractor
  ```

  **Download links:**
  - `vggt.safetensors`: Download `model.pt` from [Hugging Face VGGT-1B](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt) (requires access approval) and rename to `vggt.safetensors`, or convert using safetensors library
  - `vggsfm_v2_tracker.pt`: [Direct Download](https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt)
  - `dinov2_vitb14_reg4_pretrain.pth`: [Direct Download](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)
  - `aliked-n16.pth`
* **Depth Anything V2**
  Download the model checkpoint and place it in `dpv2/checkpoints/`:

  ```text
  dpv2/checkpoints/
    └── depth_anything_v2_vitl.pth          # Depth Anything V2 Large model
  ```

  **Download links:**
  - `depth_anything_v2_vitl.pth`: [Direct Download (Large)](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)

### Step 3: Run the Pipeline

Prepare your scene images (stored in `scene/images`) and execute the main script:

```bash
python pipeline.py --scene_dir path/to/your/scene
```

---

## 3. Core Features

* **VGGT-based Reconstruction**
  Leverages the Visual Geometry Grounded Transformer to improve camera pose estimation and sparse point cloud reconstruction.

* **Depth Estimation & Regularization**
  Integrates Depth Anything V2 to provide dense depth maps, improving training stability and geometric consistency.

* **Real-time Rendering**
  Generates high-quality 3D Gaussian Splatting models for fast and efficient novel view synthesis.
