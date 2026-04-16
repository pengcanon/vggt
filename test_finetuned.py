"""
Test a fine-tuned VGGT model on sample images and visualize the point cloud.
Uses step-by-step inference: aggregator -> camera_head -> depth_head -> unproject.

Usage:
    python test_finetuned.py                          # apple (default)
    python test_finetuned.py --category human_body    # human body
    python test_finetuned.py --category apple --pretrained  # pretrained model on apple
"""

import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# Per-category config: image_dir, mask_dir, checkpoint, image extension
CATEGORY_CONFIG = {
    "apple": {
        "image_dir": r"d:\GitHub\vggt\datasets\co3d\dataset\multi_view\apple\110_13051_23361\images",
        "mask_dir": r"d:\GitHub\vggt\datasets\co3d\dataset\multi_view\apple\110_13051_23361\masks",
        "checkpoint": r"d:\GitHub\vggt\training\logs\apple_finetune\ckpts\checkpoint.pt",
        "ext": "*.jpg",
    },
    "human_body": {
        "image_dir": r"d:\GitHub\vggt\datasets\human_body\human_body_00\sequence_001\images",
        "mask_dir": r"d:\GitHub\vggt\datasets\human_body\human_body_00\sequence_001\masks",
        "checkpoint": r"d:\GitHub\vggt\training\logs\human_body_finetune\ckpts\checkpoint.pt",
        "ext": "*.jpg",
        "annotation_file": r"d:\GitHub\vggt\datasets\human_body\annotations\human_body_test.jgz",
    },
}


def create_colored_point_cloud(point_map, image_path, mask_dir=None):
    """Create a colored point cloud from VGGT point map output, optionally masked."""
    import open3d as o3d

    original_img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = original_img.size

    target_size = 518
    new_width = target_size
    new_height = round(orig_height * (new_width / orig_width) / 14) * 14

    img_scaled = original_img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img_array = np.array(img_scaled).astype(np.float32) / 255.0

    # Load and resize mask if available
    obj_mask = None
    if mask_dir is not None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(mask_dir, base_name + ".png")
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            mask_scaled = mask_img.resize((new_width, new_height), Image.Resampling.NEAREST)
            obj_mask = np.array(mask_scaled).astype(np.float32) / 255.0

    if new_height > target_size:
        crop_offset_y = (new_height - target_size) // 2
        img_array = img_array[crop_offset_y:crop_offset_y + target_size, :, :]
        if obj_mask is not None:
            obj_mask = obj_mask[crop_offset_y:crop_offset_y + target_size, :]

    points = point_map.reshape(-1, 3)
    colors = img_array.reshape(-1, 3)

    valid_mask = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
    if obj_mask is not None:
        valid_mask &= obj_mask.reshape(-1) > 0.5
    points = points[valid_mask]
    colors = colors[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model instead of fine-tuned")
    parser.add_argument("--category", default="apple", choices=list(CATEGORY_CONFIG.keys()),
                        help="Category to test on (default: apple)")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample")
    parser.add_argument("--use_gt_cameras", action="store_true", help="Use ground truth cameras instead of predicted cameras")
    args = parser.parse_args()

    cfg = CATEGORY_CONFIG[args.category]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    if args.pretrained:
        # Use the full pretrained model (all heads enabled)
        checkpoint_path = r"d:\GitHub\vggt\vggt\pretrained\model.pt"
        print(f"Loading PRETRAINED model from {checkpoint_path}...")
        model = VGGT()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        # Use the fine-tuned checkpoint for the selected category
        checkpoint_path = cfg["checkpoint"]
        print(f"Loading FINE-TUNED model ({args.category}) from {checkpoint_path}...")
        model = VGGT(
            enable_camera=True,
            enable_depth=True,
            enable_point=False,
            enable_track=False,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model = model.to(device)
    print("Model loaded successfully")

    # Load sample images (pick random frames from one sequence)
    image_dir = cfg["image_dir"]
    all_images = sorted(glob.glob(os.path.join(image_dir, cfg["ext"])))
    if not all_images:
        print(f"No images found in {image_dir}")
        return
    num_frames = min(args.num_frames, len(all_images))
    indices = np.sort(np.random.choice(len(all_images), num_frames, replace=False))
    image_names = [all_images[i] for i in indices]
    print(f"Selected {len(image_names)} images from {image_dir}:")
    for name in image_names:
        print(f"  {os.path.basename(name)}")

    image_inputs = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {image_inputs.shape}")

    # Load GT cameras if requested
    gt_cameras = None
    if args.use_gt_cameras and "annotation_file" in cfg:
        import gzip
        import json
        
        print(f"Loading GT cameras from {cfg['annotation_file']}...")
        with gzip.open(cfg["annotation_file"], "rt") as f:
            anno_data = json.load(f)
            
        gt_extrinsics = []
        gt_intrinsics = []
        
        # Need to match image names to filepath in annotations
        # image_names are full absolute paths, annotations are relative to dataset root
        dataset_root = os.path.dirname(os.path.dirname(cfg["annotation_file"]))
        
        for img_path in image_names:
            rel_path = os.path.relpath(img_path, dataset_root).replace("\\", "/")
            seq_name = img_path.split(os.sep)[-3]  # e.g., sequence_001
            found = False
            
            # The structure in annotations is {sequence_name: [{filepath: ..., extri: ..., intri: ...}, ...]}
            if seq_name in anno_data:
                for frame in anno_data[seq_name]:
                    if frame["filepath"] == rel_path:
                        gt_extrinsics.append(frame["extri"])
                        gt_intrinsics.append(frame["intri"])
                        found = True
                        break
            
            # Fallback search if not found
            if not found:
                for seq_frames in anno_data.values():
                    for frame in seq_frames:
                        if frame["filepath"] == rel_path:
                            gt_extrinsics.append(frame["extri"])
                            gt_intrinsics.append(frame["intri"])
                            found = True
                            break
                            
            if not found:
                raise ValueError(f"Could not find GT cameras for {rel_path} in annotations")
                
        # Format tensors and apply VGGT camera formatting
        # VGGT expects extrinsics [B, V, 4, 4] and intrinsics [B, V, 3, 3] on the correct device
        gt_intrinsics = torch.tensor(gt_intrinsics, dtype=torch.float32, device=device).unsqueeze(0)
        
        # GT extrinsics are 3x4, extend to 4x4
        gt_ext_4x4 = []
        for ext in gt_extrinsics:
            ext_tensor = torch.tensor(ext, dtype=torch.float32, device=device)
            bottom_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)
            ext_4x4 = torch.cat([ext_tensor, bottom_row], dim=0)
            gt_ext_4x4.append(ext_4x4)
        gt_extrinsics = torch.stack(gt_ext_4x4).unsqueeze(0)
        
        gt_cameras = (gt_extrinsics, gt_intrinsics)

    # Step-by-step inference (matching notebook pattern)
    print("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            image_inputs = image_inputs[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(image_inputs)

        # Predict cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_inputs.shape[-2:])
        
        # Override with GT cameras if requested
        if gt_cameras is not None:
            extrinsic, intrinsic = gt_cameras
            print(f"Using GT cameras shape: {extrinsic.shape}, {intrinsic.shape}")

        # Predict depth maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image_inputs, ps_idx)

        # Construct 3D points from depth maps and cameras
        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0),
            extrinsic.squeeze(0),
            intrinsic.squeeze(0),
        )

    print(f"Depth map shape: {depth_map.shape}")
    print(f"Extrinsic shape: {extrinsic.shape}")
    print(f"Point map shape: {point_map_by_unprojection.shape}")

    # Show the 5 input images
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(image_names), figsize=(4 * len(image_names), 4))
    for idx, img_path in enumerate(image_names):
        img = Image.open(img_path).convert("RGB")
        axes[idx].imshow(img)
        axes[idx].set_title(os.path.basename(img_path), fontsize=9)
        axes[idx].axis("off")
    model_label = "Pretrained" if args.pretrained else "Fine-tuned"
    fig.suptitle(f"Input {args.category} Images ({model_label})", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # Visualize with Open3D
    import open3d as o3d

    mask_dir = cfg["mask_dir"]
    if not os.path.isdir(mask_dir):
        print(f"Warning: mask directory not found at {mask_dir}, generating PCD without masks")
        mask_dir = None

    all_pcds = []
    for idx, img_path in enumerate(image_names):
        pcd = create_colored_point_cloud(point_map_by_unprojection[idx], img_path, mask_dir=mask_dir)
        all_pcds.append(pcd)
        print(f"Image {idx} ({os.path.basename(img_path)}): {len(pcd.points)} valid points")

    o3d.visualization.draw_geometries(all_pcds, window_name=f"{args.category} - {model_label} VGGT")


if __name__ == "__main__":
    main()
