import os
import glob
import gzip
import json
import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from prepare_human_body import convert_pt3d_to_opencv

IMAGE_DIR        = r"d:\GitHub\vggt\datasets\human_body\human_body_00\sequence_001\images"
RAW_ANNO_FILE    = r"d:\GitHub\vggt\datasets\human_body\human_body_00\frame_annotations.jgz"
DATASET_ROOT     = r"d:\GitHub\vggt\datasets\human_body"
PRETRAINED_CKPT  = r"d:\GitHub\vggt\vggt\pretrained\model.pt"

device = "cuda"
dtype = torch.bfloat16

model = VGGT()
model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location="cpu"))
model.eval().to(device)

all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

# Test with 6 evenly spaced cameras (one from each of the 5 rings)
indices = np.linspace(0, len(all_images) - 1, 6, dtype=int)
image_names = [all_images[i] for i in indices]

with gzip.open(RAW_ANNO_FILE, "rt") as f:
    all_frames = json.load(f)
frame_lookup = {fr["image"]["path"].replace("\\", "/"): fr for fr in all_frames}

gt_ext_list = []
for img_path in image_names:
    rel_path = os.path.relpath(img_path, DATASET_ROOT).replace("\\", "/")
    fr = frame_lookup[rel_path]
    with Image.open(img_path) as im:
        W, H = im.size
    vp = fr["viewpoint"]
    extri_3x4, _ = convert_pt3d_to_opencv(
        vp["R"], vp["T"], vp["focal_length"], vp["principal_point"], (W, H)
    )
    gt_ext_list.append(np.array(extri_3x4, dtype=np.float32))

gt_ext_np = np.stack(gt_ext_list)
bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (6, 1, 1))
gt_ext_4x4 = np.concatenate([gt_ext_np, bottom], axis=1)
gt_ext_t = torch.tensor(gt_ext_4x4, dtype=torch.float32, device=device).unsqueeze(0)

# 1. Normalize GT cameras so the first camera is the origin
first_inv = closed_form_inverse_se3(gt_ext_t[:, 0])
gt_ext_rel = torch.matmul(gt_ext_t, first_inv.unsqueeze(1))

# Predict cameras
image_inputs = load_and_preprocess_images(image_names).to(device)
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        imgs_b = image_inputs[None]
        agg, ps_idx = model.aggregator(imgs_b)
    pose_enc = model.camera_head(agg)[-1]
    pred_ext, pred_int = pose_encoding_to_extri_intri(pose_enc, imgs_b.shape[-2:])

pred_ext_rel = pred_ext

print("Comparing Relative Rotations and Translations (Cam0 = Identity)")
for i in range(1, 6):
    pred_R = pred_ext_rel[0, i, :3, :3].cpu().numpy()
    pred_T = pred_ext_rel[0, i, :3, 3].cpu().numpy()

    gt_R = gt_ext_rel[0, i, :3, :3].cpu().numpy()
    gt_T = gt_ext_rel[0, i, :3, 3].cpu().numpy()

    # Rotation trace difference
    R_diff = pred_R @ gt_R.T
    tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(tr))
    
    print(f"\nCam {i}")
    print(f"  Rot Error: {angle:.2f} degrees")
    print(f"  Pred T: {pred_T}")
    print(f"  GT T:   {gt_T}")
    pred_norm = np.linalg.norm(pred_T)
    gt_norm = np.linalg.norm(gt_T)
    print(f"  Scale ratio (Pred/GT): {pred_norm / (gt_norm + 1e-8):.4f}")
