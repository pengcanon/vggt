import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import initialize, compose
from data.datasets.co3d import Co3dDataset

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
    print("VGGT Human Body Dataloader Test")
    print("=" * 60)

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="human_body_finetune_linux")

    # Wrap config dict as an object with attribute access for BaseDataset
    class DictConf:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, DictConf(v) if isinstance(v, dict) else v)

    common_conf = OmegaConf.to_container(cfg.data.train.common_config, resolve=True)
    common_conf_obj = DictConf(common_conf)

    dataset_cfg = cfg.data.train.dataset.dataset_configs[0]
    
    # For testing on Windows, replace WSL paths with native paths
    windows_dir = dataset_cfg.CO3D_DIR.replace("/mnt/d/", "D:/")
    windows_anno = dataset_cfg.CO3D_ANNOTATION_DIR.replace("/mnt/d/", "D:/")
    
    print(f"\n[1] Building Co3dDataset (train split)...")
    print(f"  DIR: {windows_dir}")
    print(f"  ANNO: {windows_anno}")
    print(f"  CATEGORIES: {list(dataset_cfg.categories)}")
    
    train_ds = Co3dDataset(
        common_conf=common_conf_obj,
        split="train",
        CO3D_DIR=windows_dir,
        CO3D_ANNOTATION_DIR=windows_anno,
        categories=list(dataset_cfg.categories),
        len_train=10,
    )
    
    print(f"  Sequences loaded: {len(train_ds.sequence_list)}")
    
    if len(train_ds.sequence_list) == 0:
        print("\nERROR: No sequences were loaded!")
        return

    print(f"\n[2] Sampling a sequence...")
    # Get 4 images from sequence 0
    try:
        batch = train_ds.get_data(seq_index=0, img_per_seq=4)
        
        print("\nBatch successfully loaded! Contents:")
        for k, v in batch.items():
            if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'shape'):
                print(f"  {k}: list[{len(v)}] of arrays, shape={v[0].shape}")
            elif isinstance(v, np.ndarray):
                print(f"  {k}: ndarray {v.shape}")
            else:
                print(f"  {k}: {type(v).__name__}")
                
        print("\nGenerating Point Cloud...")
        all_pts = []
        all_cols = []
        for i in range(len(batch["images"])):
            pts = batch["world_points"][i]
            img = batch["images"][i]
            mask = batch["point_masks"][i]
            
            valid_pts = pts[mask]
            # Images in dataloader might be normalized, assume 0-1 or normalized by timm
            valid_cols = img[mask]
            # Since standard timm normalization puts values in a specific range (-2 to +2ish),
            # we might need to unnormalize or just save as is and see.
            # To be safe, if we notice values out of [0, 1] range, we can clip/scale.
            if valid_cols.max() > 2.0:
                valid_cols = valid_cols / 255.0
            elif valid_cols.min() < 0:
                # Approximate un-normalization for display (ImageNet mean/std)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                valid_cols = valid_cols * std + mean
                valid_cols = np.clip(valid_cols, 0.0, 1.0)
                
            all_pts.append(valid_pts)
            all_cols.append(valid_cols)
            
        final_pts = np.concatenate(all_pts, axis=0)
        final_cols = np.concatenate(all_cols, axis=0)
        
        out_path = os.path.join(os.path.dirname(__file__), "logs", "human_body_test.ply")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_ply(final_pts, final_cols, out_path)
                
        print("\nSUCCESS! Dataloader processed the human_body annotations, loaded images and depths, and applied masks.")
    except Exception as e:
        print(f"\nFAILED to sample a batch: {e}")

if __name__ == "__main__":
    main()
