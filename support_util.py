import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2 
import torch.nn.functional as F
import pickle
import hashlib
import json
from collections import defaultdict

def extract_masks_from_support(sam_predictor, pil_img, ref_boxes, device="cuda"):
    """
    Use SAM to extract masks from a list of bounding boxes on a support image.
    
    Args:
        sam_predictor: Initialized SAM predictor (e.g., SAM2 predictor).
        img: numpy image (H x W x 3), loaded with cv2 or PIL.
        img_path: str, image file path.
        ref_boxes: a bbox with [x1, y1, x2, y2] 
        device: torch device string.

    Returns:
        reference_data: dict with masks for the given class.
    """
    
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        sam_predictor.set_image(pil_img)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=torch.tensor(ref_boxes)[None, :],  # ensure shape (1, 4)
            multimask_output=False,
        )
      
    return masks[0]
    
def get_dinov2_features(dinov2_model, dinov2_transform, pil_img, device='cpu'):
    pil_img = resize_with_aspect_ratio(pil_img, target_long_side=630, patch_size=14)
    image_tensor = dinov2_transform(pil_img)[None].to(device)
    with torch.inference_mode():
        output = dinov2_model.get_intermediate_layers(image_tensor, n =1, reshape=True, return_class_token=True, norm=False)
        output = torch.stack([out[0] for out in output], dim=0).sum(dim=0)
        return output # Shape: (B, C, H_feat, W_feat)

def resize_with_aspect_ratio(img_pil, target_long_side=1024, patch_size=16):
    """
    Resize a PIL image to have a specific long side, maintaining aspect ratio,
    and ensure new dimensions are multiples of the patch size.
    Uses BICUBIC filter for resampling.

    Args:
        img_pil (PIL.Image): Input image.
        target_long_side (int): Desired size of the longer side.
        patch_size (int): Size of the patches, new dimensions must be multiples of this.

    Returns:
        PIL.Image: Resized image with dimensions as multiples of patch_size.
    """
    orig_width, orig_height = img_pil.size
    aspect_ratio = orig_width / orig_height

    # Calculate initial resized dimensions based on long side
    if orig_width >= orig_height:
        new_width = target_long_side
        new_height = int(target_long_side / aspect_ratio)
    else:
        new_height = target_long_side
        new_width = int(target_long_side * aspect_ratio)

    # Ensure dimensions are multiples of patch_size
    # Using floor division to guarantee we don't exceed target_long_side
    new_width = max((new_width // patch_size), 1) * patch_size
    new_height = max((new_height // patch_size),1) * patch_size

    return img_pil.resize((new_width, new_height), resample=Image.BICUBIC)

def resize_mask_to_features(mask_np, feature_map_shape):
    H_feat, W_feat = feature_map_shape[0], feature_map_shape[1]
    
    # Handle different input dimensions
    if mask_np.ndim == 3:
        # If input is 3D (e.g., batch dimension), take the first mask
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]  # Remove batch dimension
        else:
            # If multiple masks, take the first one
            mask_np = mask_np[0]
    
    # Ensure mask is 2D
    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_np.ndim}D with shape {mask_np.shape}")
    
    # cv2.resize expects dsize as (width, height), not (height, width)
    resized_mask = cv2.resize(mask_np.astype(np.float32), dsize=(W_feat, H_feat))
    return (resized_mask > 0.5).astype(np.float32)
    
def _support_sample_key(img_path, bbox):
    stat = os.stat(img_path)
    payload = {
        'image': os.path.abspath(img_path),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
        'bbox': [float(value) for value in bbox],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()


def _load_support_feature_cache(cache_path, rebuild=False):
    if not cache_path or rebuild or not os.path.exists(cache_path):
        return {}
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    features = cache.get('features', {}) if isinstance(cache, dict) else {}
    print(f"Loaded {len(features)} cached support instance features: {cache_path}")
    return features


def _save_support_feature_cache(cache_path, cached_features, cache_metadata=None):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = f"{cache_path}.tmp"
    torch.save(
        {'metadata': cache_metadata or {}, 'features': cached_features},
        temp_path,
    )
    os.replace(temp_path, cache_path)
    print(f"Saved {len(cached_features)} support instance features: {cache_path}")


def extract_support_features(
    support_data,
    sam2_predictor,
    feat_extractor_name,
    feat_extractor,
    image_transform,
    data_dir,
    device='cpu',
    instance_cache_path=None,
    rebuild_instance_cache=False,
    cache_metadata=None,
    support_box_batch_size=64,
):
    '''
    support_data: dict[class_name] = list of dict with keys:
        - 'image': image path (relative to data_dir)
        - 'bbox': list of one or more [x1, y1, x2, y2]

    Returns:
        features[class_name] = list of instance features (torch tensor)
    '''
    class_features = {cls: [] for cls in support_data}

    if feat_extractor_name == 'DINOV2':
        extractor = get_dinov2_features
    elif feat_extractor_name == 'RADIO':
        from model.radio import get_radio_features
        extractor = get_radio_features
    else:
        raise ValueError(f"Unsupported feature extractor: {feat_extractor_name}")
    if support_box_batch_size <= 0:
        raise ValueError("support_box_batch_size must be positive")
    cached_features = _load_support_feature_cache(
        instance_cache_path, rebuild=rebuild_instance_cache
    )
    pending_by_image = defaultdict(list)
    cache_hits = 0
    for cls, samples in support_data.items():
        for sample in samples:
            img_path = os.path.join(data_dir, sample['image'])
            sample_key = _support_sample_key(img_path, sample['bbox'])
            if sample_key in cached_features:
                class_features[cls].append(cached_features[sample_key])
                cache_hits += 1
            else:
                pending_by_image[img_path].append((cls, sample, sample_key))

    pending_count = sum(len(items) for items in pending_by_image.values())
    print(
        f"Support instances: {cache_hits} cache hits, "
        f"{pending_count} to compute from {len(pending_by_image)} images"
    )
    device_type = torch.device(device).type
    for image_index, (img_path, items) in enumerate(
        tqdm(pending_by_image.items(), desc='Support Images'), start=1
    ):
        pil_img = Image.open(img_path).convert('RGB')
        # Reuse the full-image feature map for every support box in this image.
        full_feat = extractor(
            feat_extractor, image_transform, pil_img, device=device
        )
        sam2_predictor.set_image(pil_img)

        for start in range(0, len(items), support_box_batch_size):
            batch_items = items[start:start + support_box_batch_size]
            batch_boxes = []
            for _, sample, _ in batch_items:
                x, y, w, h = sample['bbox']
                batch_boxes.append([x, y, x + w, y + h])

            with torch.inference_mode(), torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == 'cuda',
            ):
                masks, _, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(batch_boxes, dtype=np.float32),
                    multimask_output=False,
                )

            for (cls, _, sample_key), mask_np in zip(batch_items, masks):
                resized_mask = resize_mask_to_features(
                    mask_np, full_feat.shape[2:]
                )
                resized_mask_tensor = (
                    torch.from_numpy(resized_mask)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                valid_pixel_count = resized_mask_tensor.sum()
                if valid_pixel_count <= 0:
                    continue
                feat_vec = (
                    (full_feat * resized_mask_tensor).sum(dim=[2, 3])
                    / valid_pixel_count
                ).squeeze(0).detach().cpu()
                class_features[cls].append(feat_vec)
                cached_features[sample_key] = feat_vec

        # Checkpoint long all-shot builds so an interrupted run can resume.
        if instance_cache_path and image_index % 250 == 0:
            _save_support_feature_cache(
                instance_cache_path,
                cached_features,
                cache_metadata=cache_metadata,
            )

    if pending_count:
        _save_support_feature_cache(
            instance_cache_path, cached_features, cache_metadata=cache_metadata
        )

    features = {}
    for cls, cls_feats in class_features.items():
        if cls_feats:
            features[cls] = [torch.stack(cls_feats, dim=0).mean(dim=0)]
        else:
            print(f"[Warning] No valid features for class {cls}")
            features[cls] = []
    return features

def compute_prototype_weights(memory_bank, device):
    """
    features_per_class: dict[class_name] = list of torch.Tensor features (each list contains only one proto)
    returns: dict[class_name] = prototype tensor, list[class_name] = class_names (for backward compatibility)
    """
    proto_cls_list = []
    proto_feat = []
    for cls, feats in memory_bank.items():
        if len(feats) > 0:  
            proto = feats[0]  
            proto_feat.append(proto.to(device))
            proto_cls_list.append(cls)
        else:
            print(f"[Warning] No features for class {cls}, skipping.")
    
    # Return both the normalized features and the list of class names
    # This maintains backward compatibility with existing code
    return F.normalize(torch.stack(proto_feat, dim=1), dim=0), proto_cls_list
