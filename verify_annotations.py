import json, gzip, numpy as np
from PIL import Image
import os

from prepare_human_body import convert_pt3d_to_opencv

vggt_anno_file = "datasets/co3d/annotations/apple_test.jgz"
co3d_anno_path = "datasets/co3d/dataset/multi_view/apple/frame_annotations.jgz"
dataset_root = "datasets/co3d/dataset/multi_view"

with gzip.open(vggt_anno_file, "rt") as f:
    vggt_data = json.load(f)

with gzip.open(co3d_anno_path, "rt") as f:
    pt3d_data = json.load(f)

# Put VGGT data into a flat dict by filepath for easy lookup
vggt_flat = {}
for seq, frames in vggt_data.items():
    for f in frames:
        vggt_flat[f["filepath"]] = f

print(f"Loaded {len(vggt_flat)} VGGT frames for comparison.")

checked = 0
errors = 0

for d in pt3d_data:
    filepath = d["image"]["path"].replace("\\", "/")
    if filepath not in vggt_flat:
        continue
        
    vggt_frame = vggt_flat[filepath]
    
    full_img_path = os.path.join(dataset_root, filepath)
    with Image.open(full_img_path) as img:
        W, H = img.size
        
    vp = d["viewpoint"]
    my_extri, my_intri = convert_pt3d_to_opencv(
        vp["R"], vp["T"], 
        vp["focal_length"], vp["principal_point"], 
        (W, H)
    )
    
    vggt_extri = np.array(vggt_frame["extri"])
    vggt_intri = np.array(vggt_frame["intri"])
    my_extri = np.array(my_extri)
    my_intri = np.array(my_intri)
    
    diff_extri = np.abs(vggt_extri - my_extri).max()
    diff_intri = np.abs(vggt_intri - my_intri).max()
    
    if diff_extri > 1e-4 or diff_intri > 1.0:
        errors += 1
        if errors == 1:
            print("Mismatch found on:", filepath)
            print("W, H =", W, H)
            print("My Extri:\n", my_extri)
            print("VGGT Extri:\n", vggt_extri)
            print("My Intri:\n", my_intri)
            print("VGGT Intri:\n", vggt_intri)
            print("---")
            
    checked += 1

print(f"Checked {checked} frames. Errors found: {errors}")
