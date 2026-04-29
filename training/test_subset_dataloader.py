import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import initialize, compose
import logging

NUM_BATCHES_TO_TEST = 3
IMG_PER_SEQ = 8   

def save_ply(points, colors, filename):
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
    except Exception as e:
        print(f"  Export failed: {e}")

def main():
    print("=" * 60)
    print("VGGT CO3D Subset Dataloader Test")
    print("=" * 60)

    from data.datasets.co3d import Co3dDataset

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="co3d_subset_finetune_linux")

    train_ds_cfg = cfg.data.train.dataset.dataset_configs[0]
    common_conf = OmegaConf.to_object(cfg.data.train.common_config)

    class DictConf:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, DictConf(v) if isinstance(v, dict) else v)

    common_conf_obj = DictConf(common_conf)

    # Use the directories from the config. 
    # For testing on Windows natively, if the paths in config are ../, resolve relative to current dir
    CO3D_DIR = train_ds_cfg.CO3D_DIR
    CO3D_ANNOTATION_DIR = train_ds_cfg.CO3D_ANNOTATION_DIR
    categories = train_ds_cfg.get("categories", None)
    if categories is not None:
        categories = list(categories)

    # Resolve relative paths relative to training/ directory where this script runs from
    base_dir = os.path.dirname(__file__)
    if CO3D_DIR.startswith("../"):
        # Resolve path
        CO3D_DIR = os.path.normpath(os.path.join(base_dir, CO3D_DIR))
    if CO3D_ANNOTATION_DIR.startswith("../"):
        CO3D_ANNOTATION_DIR = os.path.normpath(os.path.join(base_dir, CO3D_ANNOTATION_DIR))

    print(f"\n[1] Building Co3dDataset (train split)...")
    print(f"  Using CO3D_DIR: {CO3D_DIR}")
    print(f"  Using Annotations: {CO3D_ANNOTATION_DIR}")
    print(f"  Categories: {categories}")

    train_ds = Co3dDataset(
        common_conf=common_conf_obj,
        split="train",
        CO3D_DIR=CO3D_DIR,
        CO3D_ANNOTATION_DIR=CO3D_ANNOTATION_DIR,
        categories=categories,
        len_train=10000,
    )
    print(f"  Sequences loaded: {train_ds.sequence_list_len}")
    print(f"  Dataset total samples mapping length: {len(train_ds)}")

    import random
    test_indices = random.sample(range(train_ds.sequence_list_len), min(NUM_BATCHES_TO_TEST, train_ds.sequence_list_len))
    
    for count, i in enumerate(test_indices):
        print(f"\n  --- Sequence index {i} ({train_ds.sequence_list[i]}) ---")
        batch = train_ds.get_data(seq_index=i, img_per_seq=IMG_PER_SEQ)

        assert "images" in batch and len(batch["images"]) == IMG_PER_SEQ
        assert "extrinsics" in batch and len(batch["extrinsics"]) == IMG_PER_SEQ
        
        if count == 0:
            ply_path = os.path.join(base_dir, "debug_subset_batch0.ply")
            world_pts = np.stack(batch["world_points"]).reshape(-1, 3)
            colors = np.stack([img.transpose(1, 2, 0) if img.ndim == 3 else img
                               for img in batch["images"]]).reshape(-1, 3)
            # Normalize colors range for visualization
            colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
            save_ply(world_pts, colors, ply_path)
            
    print("\n" + "=" * 60)
    print("Dataloader text PASSED. Check debug_subset_batch0.ply for point cloud visual!")
    print("=" * 60)

if __name__ == "__main__":
    main()
