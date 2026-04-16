"""
Dataloader test script for VGGT CO3D apple fine-tuning.
Run from the training/ directory:
    python test_dataloader.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# Add repo root so 'vggt' package is importable when running from training/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import initialize, compose

CO3D_DIR = "D:/GitHub/vggt/datasets/co3d/dataset/multi_view"
CO3D_ANNOTATION_DIR = "D:/GitHub/vggt/datasets/co3d/annotations"
NUM_BATCHES_TO_TEST = 3
IMG_PER_SEQ = 8   # number of frames to sample per sequence


def save_ply(points, colors, filename):
    """Save point cloud to PLY file for visual inspection."""
    try:
        import open3d as o3d
        if torch.is_tensor(points):
            pts = points.reshape(-1, 3).cpu().numpy()
        else:
            pts = points.reshape(-1, 3)
        if torch.is_tensor(colors):
            cols = colors.reshape(-1, 3).cpu().numpy()
        else:
            cols = colors.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
        print(f"  Saved PLY: {filename}")
    except ImportError:
        print("  open3d not installed, skipping PLY export.")


def main():
    print("=" * 60)
    print("VGGT CO3D Apple Dataloader Test")
    print("=" * 60)

    # Directly instantiate Co3dDataset, bypassing DynamicTorchDataset
    # which requires torch.distributed. This tests the real data pipeline
    # (image loading, camera conversion, augmentation) without DDP overhead.
    from data.datasets.co3d import Co3dDataset
    from data.composed_dataset import ComposedDataset

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="apple_finetune")

    common_conf = OmegaConf.to_object(cfg.data.train.common_config)

    # Wrap config dict as an object with attribute access for BaseDataset
    class DictConf:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, DictConf(v) if isinstance(v, dict) else v)

    common_conf_obj = DictConf(common_conf)

    # --- Build train dataset ---
    print("\n[1] Building Co3dDataset (train split)...")
    train_ds = Co3dDataset(
        common_conf=common_conf_obj,
        split="train",
        CO3D_DIR=CO3D_DIR,
        CO3D_ANNOTATION_DIR=CO3D_ANNOTATION_DIR,
        len_train=10000,
    )
    print(f"  Sequences loaded: {train_ds.sequence_list_len}")
    print(f"  Dataset length: {len(train_ds)}")

    # --- Sample a few sequences directly ---
    print(f"\n[2] Sampling {NUM_BATCHES_TO_TEST} sequences...")
    for i in range(NUM_BATCHES_TO_TEST):
        print(f"\n  --- Sequence {i} ({train_ds.sequence_list[i]}) ---")
        batch = train_ds.get_data(seq_index=i, img_per_seq=IMG_PER_SEQ)

        for key, val in batch.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], np.ndarray):
                print(f"    {key}: list[{len(val)}] of ndarray {val[0].shape}")
            elif isinstance(val, np.ndarray):
                print(f"    {key}: ndarray {val.shape}")
            elif isinstance(val, list):
                print(f"    {key}: list of len={len(val)}, type={type(val[0]).__name__ if val else 'empty'}")
            else:
                print(f"    {key}: {val}")

        # Sanity checks
        print("  Sanity checks:")
        assert "images" in batch and len(batch["images"]) == IMG_PER_SEQ
        assert "extrinsics" in batch and len(batch["extrinsics"]) == IMG_PER_SEQ
        assert "intrinsics" in batch and len(batch["intrinsics"]) == IMG_PER_SEQ
        assert "world_points" in batch
        assert batch["extrinsics"][0].shape == (3, 4), f"Bad extrinsic shape: {batch['extrinsics'][0].shape}"
        assert batch["intrinsics"][0].shape == (3, 3), f"Bad intrinsic shape: {batch['intrinsics'][0].shape}"
        print("    All checks passed.")

        # Save PLY for first batch
        if i == 0:
            ply_path = os.path.join(os.path.dirname(__file__), "debug_batch0.ply")
            print(f"  Saving PLY to {ply_path}...")
            world_pts = np.stack(batch["world_points"]).reshape(-1, 3)
            colors = np.stack([img.transpose(1, 2, 0) if img.ndim == 3 else img
                               for img in batch["images"]]).reshape(-1, 3)
            # images are in [-1,1] or [0,1] — normalize to [0,1]
            colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
            save_ply(world_pts, colors, ply_path)

    # --- Val dataset ---
    print("\n[3] Building Co3dDataset (test split)...")
    val_ds = Co3dDataset(
        common_conf=common_conf_obj,
        split="test",
        CO3D_DIR=CO3D_DIR,
        CO3D_ANNOTATION_DIR=CO3D_ANNOTATION_DIR,
        len_test=1000,
    )
    print(f"  Val sequences loaded: {val_ds.sequence_list_len}")
    val_batch = val_ds.get_data(seq_index=0, img_per_seq=4)
    print(f"  Val sample images: {len(val_batch['images'])}, shape: {val_batch['images'][0].shape}")

    print("\n" + "=" * 60)
    print("Dataloader test PASSED. Ready to train!")
    print("=" * 60)
    print("\nTo start fine-tuning, run from the training/ directory:")
    print("  torchrun --nproc_per_node=1 launch.py --config apple_finetune")


if __name__ == "__main__":
    main()



def save_ply(points, colors, filename):
    """Save point cloud to PLY file for visual inspection."""
    try:
        import open3d as o3d
        if torch.is_tensor(points):
            pts = points.reshape(-1, 3).cpu().numpy()
        else:
            pts = points.reshape(-1, 3)
        if torch.is_tensor(colors):
            cols = colors.reshape(-1, 3).cpu().numpy()
        else:
            cols = colors.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
        print(f"  Saved PLY: {filename}")
    except ImportError:
        print("  open3d not installed, skipping PLY export.")


if __name__ == "__main__":
    main()
