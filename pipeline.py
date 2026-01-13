import argparse
import os
import sys

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from pathlib import Path
import cv2
import numpy as np
import matplotlib
# Add project root to Python path so vggt module can be imported
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vggt.demo_colmap import demo_fn
from dpv2.depth_anything_v2.dpt import DepthAnythingV2

def run_vggt_colmap(
    scene_dir: str,
    seed: int = 42,
    use_ba: bool = True,
    max_reproj_error: float = 8.0,
    shared_camera: bool = False,
    camera_type: str = "SIMPLE_PINHOLE",
    vis_thresh: float = 0.2,
    query_frame_num: int = 8,
    max_query_pts: int = 2048,
    fine_tracking: bool = True,
    conf_thres_value: float = 5.0,
):
    """
    Run VGGT COLMAP reconstruction in pipeline
    
    Args:
        scene_dir: Scene directory path, should contain images/ subfolder
        seed: Random seed
        use_ba: Whether to use Bundle Adjustment
        max_reproj_error: Maximum reprojection error for BA
        shared_camera: Whether all images share the same camera parameters
        camera_type: Camera type
        vis_thresh: Visibility threshold for tracks
        query_frame_num: Number of frames to query
        max_query_pts: Maximum number of query points
        fine_tracking: Whether to use fine tracking
        conf_thres_value: Confidence threshold value for depth filtering (without BA)
    """
    args = argparse.Namespace(
        scene_dir=scene_dir,
        seed=seed,
        use_ba=use_ba,
        max_reproj_error=max_reproj_error,
        shared_camera=shared_camera,
        camera_type=camera_type,
        vis_thresh=vis_thresh,
        query_frame_num=query_frame_num,
        max_query_pts=max_query_pts,
        fine_tracking=fine_tracking,
        conf_thres_value=conf_thres_value,
    )
    
    scene_dir = os.path.abspath(scene_dir)
    
    if not os.path.exists(scene_dir):
        raise ValueError(f"Scene directory does not exist: {scene_dir}")
    
    image_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(image_dir):
        raise ValueError(f"images/ subfolder not found in scene directory: {image_dir}")
    
    args.scene_dir = scene_dir
    
    original_cwd = os.getcwd()
    vggt_dir = project_root / "vggt"
    
    try:
        os.chdir(vggt_dir)
        
        import torch
        with torch.no_grad():
            result = demo_fn(args)
    finally:
        os.chdir(original_cwd)
    
    return result

def run_dpv2(
    scene_dir: str,
    input_size: int = 518,
    encoder: str = 'vitl',
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
):
    """
    Run Depth Anything V2 to generate depth maps.
    
    The depth maps are saved as PNG files. Use utils/make_depth_scale.py 
    to compute scale and offset for 3DGS training.
    
    Args:
        scene_dir: Scene directory path, should contain images/ subfolder
        input_size: Input image size for depth model
        encoder: Model encoder type ('vits', 'vitb', 'vitl', 'vitg')
        DEVICE: Device to run inference on
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'dpv2/checkpoints/depth_anything_v2_{encoder}.pth'))
    depth_anything = depth_anything.to(DEVICE).eval()

    scene_dir_root = Path(scene_dir)
    images_dir = scene_dir_root / "images"
    depth_dir = scene_dir_root / "depth" 
    depth_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir = scene_dir_root / "confidence"
    confidence_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(ext)))

    print(f"Processing {len(image_files)} images for depth estimation...")

    with torch.no_grad():
        for img_path in image_files:
            try:
                raw_image = cv2.imread(str(img_path))
                if raw_image is None: 
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                
                depth1 = depth_anything.infer_image(raw_image, input_size)
                depth2 = depth_anything.infer_image(cv2.flip(raw_image, 1), input_size)
                depth2 = cv2.flip(depth2, 1)
                diff = np.abs(depth1 - depth2)
                confidence = np.exp(-diff * 5.0)
                
                
                invalid_mask = (depth1 <= 0) | ~np.isfinite(depth1)
                depth1[invalid_mask] = 0.0
                
                
                depth_normalized = depth1.astype(np.float32)
                if depth_normalized.max() > 0:  
                    depth_normalized = (depth_normalized / depth_normalized.max() * 65535).astype(np.uint16)
                else:
                    depth_normalized = depth_normalized.astype(np.uint16)
                
            
                save_path = depth_dir / f"{img_path.stem}.png"
                cv2.imwrite(str(save_path), depth_normalized)
                print(f"Saved depth map: {save_path}")
                
                confidence_normalized = confidence.astype(np.float32)
                confidence_normalized = np.clip(confidence_normalized, 0.0, 1.0)
                confidence_normalized = (confidence_normalized * 65535).astype(np.uint16)
                
                
                conf_save_path = confidence_dir / img_path.name.replace(img_path.suffix, ".png")
                cv2.imwrite(str(conf_save_path), confidence_normalized)
                print(f"Saved confidence map: {conf_save_path}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    del depth_anything
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Depth estimation complete. Depth maps saved to: {depth_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--use_ba", action="store_true", default=True)
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera", action="store_true", default=False)
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--max_query_pts", type=int, default=2048)
    parser.add_argument("--fine_tracking", action="store_true", default=True)
    args = parser.parse_args()
    scene_directory = args.scene_dir
    use_ba = args.use_ba
    
    run_vggt_colmap(
        scene_dir=scene_directory,
        use_ba=True,
    )

    run_dpv2(
        scene_dir=scene_directory,
    )