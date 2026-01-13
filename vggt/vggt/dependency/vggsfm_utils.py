
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from lightglue import ALIKED, SIFT, SuperPoint

from .vggsfm_tracker import TrackerPredictor

# Suppress verbose logging from dependencies
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Constants
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def build_vggsfm_tracker(model_path=None):
    """
    Build and initialize the VGGSfM tracker.

    Args:
        model_path: Path to the model weights file. If None, weights are downloaded from HuggingFace.

    Returns:
        Initialized tracker model in eval mode.
    """
    tracker = TrackerPredictor()
    
    # Use provided model_path if given
    if model_path is not None:
        if os.path.exists(model_path):
            print(f"Loading tracker from: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            tracker.load_state_dict(state_dict)
            tracker.eval()
            return tracker
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Try to find checkpoint relative to this file's location
    # __file__ is at: vggt/vggt/dependency/vggsfm_utils.py
    # We need to go up 3 levels to reach vggt/ directory
    vggt_dir = Path(__file__).parent.parent.parent
    local_path = vggt_dir / "checkpoints" / "vggsfm_v2_tracker.pt"
    
    if os.path.exists(local_path):
        print(f"Loading tracker from local path: {local_path}")
        try:
            state_dict = torch.load(local_path, map_location='cpu')
            tracker.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading local checkpoint: {e}")
            print(f"Attempting to download from HuggingFace...")
            default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
            try:
                tracker.load_state_dict(torch.hub.load_state_dict_from_url(default_url))
            except Exception as download_error:
                raise RuntimeError(
                    f"Failed to load tracker checkpoint. "
                    f"Local file exists but failed to load: {e}. "
                    f"Download also failed: {download_error}. "
                    f"Please check your network connection or manually download the checkpoint."
                )
    else:
        print(f"Local checkpoint not found at {local_path}, downloading from HuggingFace...")
        default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
        try:
            tracker.load_state_dict(torch.hub.load_state_dict_from_url(default_url))
        except Exception as download_error:
            raise RuntimeError(
                f"Failed to download tracker checkpoint from {default_url}: {download_error}. "
                f"Please check your network connection or manually download the checkpoint to {local_path}"
            )
    
    tracker.eval()
    return tracker


def generate_rank_by_dino(
    images, query_frame_num, image_size=336, model_name="dinov2_vitb14_reg", device="cuda", spatial_similarity=False
):
    """
    Generate a ranking of frames using DINO ViT features.

    Args:
        images: Tensor of shape (S, 3, H, W) with values in range [0, 1]
        query_frame_num: Number of frames to select
        image_size: Size to resize images to before processing
        model_name: Name of the DINO model to use
        device: Device to run the model on
        spatial_similarity: Whether to use spatial token similarity or CLS token similarity

    Returns:
        List of frame indices ranked by their representativeness
    """
    # Resize images to the target size
    images = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)

    current_dir = Path(__file__).parent
    vggt_root = current_dir.parent.parent
    
    dino_repo_path = vggt_root / "dinov2"
    checkpoint_path = vggt_root / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"

    print(f"local: {dino_repo_path}")
    
    if os.path.exists(dino_repo_path):
        try:
            dino_v2_model = torch.hub.load(str(dino_repo_path), model_name, source="local", pretrained=False)
            
            if os.path.exists(checkpoint_path):
                print(f"local success: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                dino_v2_model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
                
        except Exception as e:
            print(f"local load failed: {e}")
            raise e
    else:
        raise FileNotFoundError(f"dino repo not found: {dino_repo_path}")

    dino_v2_model.eval()
    dino_v2_model = dino_v2_model.to(device)

    # Normalize images using ResNet normalization
    resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
    resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
    images_resnet_norm = (images - resnet_mean) / resnet_std

    with torch.no_grad():
        frame_feat = dino_v2_model(images_resnet_norm, is_training=True)

    # Process features based on similarity type
    if spatial_similarity:
        frame_feat = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

        # Compute the similarity matrix
        frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
        similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat = frame_feat["x_norm_clstoken"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        similarity_matrix = torch.mm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))

    distance_matrix = 100 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()

    # Conduct FPS sampling starting from the most common frame
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    # Clean up all tensors and models to free memory
    del frame_feat, frame_feat_norm, similarity_matrix, distance_matrix
    del dino_v2_model
    torch.cuda.empty_cache()

    return fps_idx


def farthest_point_sampling(distance_matrix, num_samples, most_common_frame_index=0):
    """
    Farthest point sampling algorithm to select diverse frames.

    Args:
        distance_matrix: Matrix of distances between frames
        num_samples: Number of frames to select
        most_common_frame_index: Index of the first frame to select

    Returns:
        List of selected frame indices
    """
    distance_matrix = distance_matrix.clamp(min=0)
    N = distance_matrix.size(0)

    # Initialize with the most common frame
    selected_indices = [most_common_frame_index]
    check_distances = distance_matrix[selected_indices]

    while len(selected_indices) < num_samples:
        # Find the farthest point from the current set of selected points
        farthest_point = torch.argmax(check_distances)
        selected_indices.append(farthest_point.item())

        check_distances = distance_matrix[farthest_point]
        # Mark already selected points to avoid selecting them again
        check_distances[selected_indices] = 0

        # Break if all points have been selected
        if len(selected_indices) == N:
            break

    return selected_indices


def calculate_index_mappings(query_index, S, device=None):
    """
    Construct an order that switches [query_index] and [0]
    so that the content of query_index would be placed at [0].

    Args:
        query_index: Index to swap with 0
        S: Total number of elements
        device: Device to place the tensor on

    Returns:
        Tensor of indices with the swapped order
    """
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def switch_tensor_order(tensors, order, dim=1):
    """
    Reorder tensors along a specific dimension according to the given order.

    Args:
        tensors: List of tensors to reorder
        order: Tensor of indices specifying the new order
        dim: Dimension along which to reorder

    Returns:
        List of reordered tensors
    """
    return [torch.index_select(tensor, dim, order) if tensor is not None else None for tensor in tensors]


def initialize_feature_extractors(max_query_num, det_thres=0.005, extractor_method="aliked", device="cuda"):
    """
    Initialize feature extractors that can be reused based on a method string.

    Args:
        max_query_num: Maximum number of keypoints to extract
        det_thres: Detection threshold for keypoint extraction
        extractor_method: String specifying which extractors to use (e.g., "aliked", "sp+sift", "aliked+sp+sift")
        device: Device to run extraction on

    Returns:
        Dictionary of initialized extractors
    """
    extractors = {}
    methods = extractor_method.lower().split("+")
    
    # Get checkpoint path for ALIKED
    vggt_dir = Path(__file__).parent.parent.parent
    aliked_checkpoint_path = vggt_dir / "checkpoints" / "aliked-n16.pth"

    for method in methods:
        method = method.strip()
        if method == "aliked":
            vggt_dir = Path(__file__).parent.parent.parent
            aliked_file = vggt_dir / "checkpoints" / "aliked-n16.pth"
            
            import torch.hub
            original_load_url = torch.hub.load_state_dict_from_url

            def mocked_load_url(url, **kwargs):
                if "aliked" in url.lower() and aliked_file.exists():
                    return torch.load(aliked_file, map_location='cpu')
                return original_load_url(url, **kwargs)

            torch.hub.load_state_dict_from_url = mocked_load_url

            try:
                aliked_extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
                extractors["aliked"] = aliked_extractor.to(device).eval()
            finally:
                torch.hub.load_state_dict_from_url = original_load_url

    if not extractors:
        print(f"Warning: No valid extractors found in '{extractor_method}'. Using ALIKED by default.")
        aliked_extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
        
        # Try to load from local checkpoint if available
        if os.path.exists(aliked_checkpoint_path):
            try:
                print(f"Loading ALIKED weights from local checkpoint: {aliked_checkpoint_path}")
                local_state_dict = torch.load(aliked_checkpoint_path, map_location='cpu')
                aliked_extractor.load_state_dict(local_state_dict, strict=True)
                print(f"Successfully loaded ALIKED weights from local checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load ALIKED from local checkpoint: {e}")
                print(f"Using weights downloaded by ALIKED initialization")
        else:
            print(f"Local ALIKED checkpoint not found at {aliked_checkpoint_path}, using downloaded weights")
        
        extractors["aliked"] = aliked_extractor.to(device).eval()

    return extractors


def extract_keypoints(query_image, extractors, round_keypoints=True):
    """
    Extract keypoints using pre-initialized feature extractors.

    Args:
        query_image: Input image tensor (3xHxW, range [0, 1])
        extractors: Dictionary of initialized extractors

    Returns:
        Tensor of keypoint coordinates (1xNx2)
    """
    query_points = None

    with torch.no_grad():
        for extractor_name, extractor in extractors.items():
            query_points_data = extractor.extract(query_image, invalid_mask=None)
            extractor_points = query_points_data["keypoints"]
            if round_keypoints:
                extractor_points = extractor_points.round()

            if query_points is not None:
                query_points = torch.cat([query_points, extractor_points], dim=1)
            else:
                query_points = extractor_points

    return query_points


def predict_tracks_in_chunks(
    track_predictor, images_feed, query_points_list, fmaps_feed, fine_tracking, num_splits=None, fine_chunk=40960
):
    """
    Process a list of query points to avoid memory issues.

    Args:
        track_predictor (object): The track predictor object used for predicting tracks.
        images_feed (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
        query_points_list (list or tuple): A list/tuple of tensors, each of shape (B, Ni, 2) representing chunks of query points.
        fmaps_feed (torch.Tensor): A tensor of feature maps for the tracker.
        fine_tracking (bool): Whether to perform fine tracking.
        num_splits (int, optional): Ignored when query_points_list is provided. Kept for backward compatibility.

    Returns:
        tuple: A tuple containing the concatenated predicted tracks, visibility, and scores.
    """
    # If query_points_list is not a list or tuple but a single tensor, handle it like the old version for backward compatibility
    if not isinstance(query_points_list, (list, tuple)):
        query_points = query_points_list
        if num_splits is None:
            num_splits = 1
        query_points_list = torch.chunk(query_points, num_splits, dim=1)

    # Ensure query_points_list is a list for iteration (as torch.chunk returns a tuple)
    if isinstance(query_points_list, tuple):
        query_points_list = list(query_points_list)

    fine_pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    for split_points in query_points_list:
        # Feed into track predictor for each split
        fine_pred_track, _, pred_vis, pred_score = track_predictor(
            images_feed, split_points, fmaps=fmaps_feed, fine_tracking=fine_tracking, fine_chunk=fine_chunk
        )
        fine_pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    # Concatenate the results from all splits
    fine_pred_track = torch.cat(fine_pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)

    if pred_score is not None:
        pred_score = torch.cat(pred_score_list, dim=2)
    else:
        pred_score = None

    return fine_pred_track, pred_vis, pred_score
