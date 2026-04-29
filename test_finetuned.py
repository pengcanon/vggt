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
        "image_dir": "datasets/co3d/dataset/multi_view/apple/110_13051_23361/images",
        "mask_dir": "datasets/co3d/dataset/multi_view/apple/110_13051_23361/masks",
        "checkpoint": "training/logs/co3d_subset_finetune/ckpts/checkpoint.pt",
        "ext": "*.jpg",
    },
    "human_body": {
        "image_dir": "datasets/human_body/human_body_03/sequence_001/images",
        "mask_dir": "datasets/human_body/human_body_03/sequence_001/masks",
        "checkpoint": "training/logs/human_body_finetune/ckpts/checkpoint.pt",
        "ext": "*.jpg",
        # Raw OpenCV frame_annotations.jgz for the sequence being tested
        "raw_annotation_file": "datasets/human_body/human_body_03/frame_annotations.jgz",
        "dataset_root": "datasets/human_body",
        # Rig layout: 5 horizontal rings x 80 cameras each, ordered ring-by-ring in the annotation file
        "ring_size": 80,
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
    parser.add_argument("--num_frames", type=int, default=12, help="Number of frames to sample")
    parser.add_argument("--use_gt_cameras", action="store_true", help="Use ground truth cameras instead of predicted cameras")
    parser.add_argument("--gt_source", default="raw", choices=["raw", "generated"],
                        help="GT camera source: 'raw' uses frame_annotations.jgz (OpenCV w2c), "
                             "'generated' uses pre-converted annotation file from prepare_human_body.py")
    parser.add_argument("--fixed", action="store_true",
                        help="Select frames at fixed evenly-spaced indices (no randomness) for reproducible comparison")
    parser.add_argument("--ring", type=str, default="0",
                        help="Which ring(s) to sample from. Use an integer for a single ring (e.g. 0, 1, 2) "
                             "or 'all' to sample one camera from every elevation ring.")
    args = parser.parse_args()

    cfg = CATEGORY_CONFIG[args.category]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    if args.pretrained:
        # Use the full pretrained model (all heads enabled)
        checkpoint_path = "vggt/pretrained/model.pt"
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

    # Load sample images
    image_dir = cfg["image_dir"]
    all_images = sorted(glob.glob(os.path.join(image_dir, cfg["ext"])))
    if not all_images:
        print(f"No images found in {image_dir}")
        return
    num_frames = min(args.num_frames, len(all_images))
    # Parse --ring: integer or "all"
    if args.ring.lower() == "all":
        ring_selection = "all"
    else:
        try:
            ring_selection = int(args.ring)
        except ValueError:
            raise ValueError(f"--ring must be an integer or 'all', got '{args.ring}'")

    if args.fixed and ring_selection == "all" and "ring_size" in cfg:
        # Pick one camera per elevation ring, all at the same horizontal angle
        ring_size = cfg["ring_size"]
        total_rings = len(all_images) // ring_size
        horizontal_idx = ring_size // 2
        image_names = [all_images[r * ring_size + horizontal_idx] for r in range(total_rings)]
        print(f"[fixed all-rings] Picked camera index {horizontal_idx} from each of {total_rings} rings")
    elif args.fixed:
        # Evenly-spaced indices within a single ring
        ring_idx = ring_selection if isinstance(ring_selection, int) else 0
        ring_size = cfg.get("ring_size", len(all_images))
        ring_start = ring_idx * ring_size
        ring_end = min(ring_start + ring_size, len(all_images))
        ring_images = all_images[ring_start:ring_end]
        num_in_ring = min(num_frames, len(ring_images))
        indices_in_ring = np.linspace(0, len(ring_images) - 1, num_in_ring, dtype=int)
        image_names = [ring_images[i] for i in indices_in_ring]
        print(f"[fixed] Ring {ring_idx} (frames {ring_start}-{ring_end-1}), selecting {num_in_ring} evenly-spaced cameras")
    else:
        indices = np.sort(np.random.choice(len(all_images), num_frames, replace=False))
        image_names = [all_images[i] for i in indices]
    print(f"Selected {len(image_names)} images from {image_dir}:")
    for name in image_names:
        print(f"  {os.path.basename(name)}")

    image_inputs = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {image_inputs.shape}")

    # Load GT cameras if requested
    gt_extrinsics_3x4 = None
    gt_intrinsics_518 = None
    if args.use_gt_cameras:
        import gzip
        import json

        if args.gt_source == "raw" and "raw_annotation_file" in cfg:
            # Raw annotations are in OpenCV w2c format: R (3x3), T (3,)
            print(f"Loading raw GT cameras from {cfg['raw_annotation_file']}...")
            with gzip.open(cfg["raw_annotation_file"], "rt") as f:
                all_frames = json.load(f)

            dataset_root = cfg["dataset_root"]
            frame_lookup = {fr["image"]["path"].replace("\\", "/"): fr for fr in all_frames}

            gt_ext_list = []
            gt_int_list = []
            for img_path in image_names:
                rel_path = os.path.relpath(img_path, dataset_root).replace("\\", "/")
                if rel_path not in frame_lookup:
                    raise ValueError(f"Could not find GT cameras for {rel_path}")
                vp = frame_lookup[rel_path]["viewpoint"]
                R = np.array(vp["R"], dtype=np.float32)
                T = np.array(vp["T"], dtype=np.float32).reshape(3, 1)
                gt_ext_list.append(np.hstack((R, T)))
                # Scale intrinsics from native resolution to the 518px model input.
                # Mirrors load_and_preprocess_images (mode="crop"): width->518,
                # height to nearest multiple of 14, center-crop if height > 518.
                # Intrinsics are already in OpenCV pixel units at native resolution.
                with Image.open(img_path) as img:
                    W_orig, H_orig = img.size
                fx, fy = vp["focal_length"]     # pixels at native resolution
                cx, cy = vp["principal_point"]  # pixels at native resolution
                H_res  = round(H_orig * (518 / W_orig) / 14) * 14
                sw = 518 / W_orig;  sh = H_res / H_orig
                crop_y = (H_res - 518) // 2 if H_res > 518 else 0
                gt_int_list.append(np.array([
                    [fx * sw,  0.,          cx * sw          ],
                    [0.,          fy * sh,  cy * sh - crop_y ],
                    [0.,          0.,          1.             ]
                ], dtype=np.float32))
            gt_extrinsics_3x4 = np.array(gt_ext_list, dtype=np.float32)   # (S, 3, 4)
            gt_intrinsics_518 = np.array(gt_int_list, dtype=np.float32)    # (S, 3, 3)

        elif args.gt_source == "generated":
            dataset_root = cfg["dataset_root"]
            anno_dir = os.path.join(dataset_root, "annotations")
            for split in ("test", "train"):
                candidate = os.path.join(anno_dir, f"human_body_{split}.jgz")
                if os.path.exists(candidate):
                    anno_file = candidate
                    break
            else:
                raise FileNotFoundError(f"No generated annotation file found in {anno_dir}")

            print(f"Loading generated GT cameras from {anno_file}...")
            with gzip.open(anno_file, "rt") as f:
                anno_data = json.load(f)

            frame_lookup = {}
            for seq_frames in anno_data.values():
                for frame in seq_frames:
                    frame_lookup[frame["filepath"]] = frame

            gt_ext_list = []
            for img_path in image_names:
                rel_path = os.path.relpath(img_path, dataset_root).replace("\\", "/")
                if rel_path not in frame_lookup:
                    raise ValueError(f"Could not find GT cameras for {rel_path} in {anno_file}")
                gt_ext_list.append(frame_lookup[rel_path]["extri"])  # already OpenCV 3x4
            gt_extrinsics_3x4 = np.array(gt_ext_list, dtype=np.float32)  # (S, 3, 4)

    # Step-by-step inference (matching notebook pattern)
    print("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            image_inputs = image_inputs[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(image_inputs)

        # Predict cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_inputs.shape[-2:])

        # Override cameras with GT if requested.
        # Strategy: keep GT cameras in metric space (normalised to cam-0 only);
        # scale predicted depth UP by avg_scale so depth and cameras are in the same units.
        # avg_scale = mean(||t_gt_rel||) / mean(||t_pred||)  — the same factor VGGT divided
        # depths and translations by during training normalisation.
        avg_scale = None
        if gt_extrinsics_3x4 is not None:
            from vggt.utils.geometry import closed_form_inverse_se3

            S = gt_extrinsics_3x4.shape[0]
            bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (S, 1, 1))
            gt_ext_4x4 = torch.tensor(
                np.concatenate([gt_extrinsics_3x4, bottom], axis=1),
                dtype=torch.float32, device=device
            ).unsqueeze(0)                                                    # (1, S, 4, 4)
            first_inv = closed_form_inverse_se3(gt_ext_4x4[:, 0])
            gt_ext_rel = torch.matmul(gt_ext_4x4, first_inv.unsqueeze(1))    # (1, S, 4, 4)

            pred_t = torch.linalg.norm(extrinsic[:, 1:, :3, 3], dim=-1)
            gt_t   = torch.linalg.norm(gt_ext_rel[:, 1:, :3, 3], dim=-1)
            avg_scale = (gt_t.mean() / (pred_t.mean() + 1e-8)).item()

            extrinsic = gt_ext_rel   # metric, cam-0 normalised
            intrinsic = torch.tensor(gt_intrinsics_518, dtype=torch.float32, device=device).unsqueeze(0)
            print(f"GT cameras: cam-0 normalised, metric. avg_scale={avg_scale:.4f}")

        # Predict depth maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image_inputs, ps_idx)

        # Scale depth from VGGT normalised units up to metric to match GT cameras
        if avg_scale is not None:
            depth_map = depth_map * avg_scale

        # Unproject using final extrinsics + intrinsics
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

    fig, axes = plt.subplots(1, len(image_names), figsize=(4 * len(image_names), 4), squeeze=False)
    for idx, img_path in enumerate(image_names):
        img = Image.open(img_path).convert("RGB")
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(os.path.basename(img_path), fontsize=9)
        axes[0, idx].axis("off")
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
