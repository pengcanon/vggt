import os
import glob
import json
import gzip
import numpy as np
from PIL import Image

def convert_pt3d_to_opencv(R_pt3d, T_pt3d, focal_length, principal_point, image_size):
    R = np.array(R_pt3d)
    T = np.array(T_pt3d)
    
    # PT3D World-to-Camera rotation and translation
    R_w2c = R.T
    T_w2c = T
    
    # PyTorch3D to OpenCV: Flip X and Y
    flip = np.array([[-1, 0, 0], 
                     [0, -1, 0], 
                     [0,  0, 1]], dtype=np.float32)
    
    R_cv = flip @ R_w2c
    T_cv = flip @ T_w2c
    
    extri = np.hstack((R_cv, T_cv.reshape(3, 1)))
    
    W, H = image_size
    s = min(H, W) / 2.0
    
    fx = focal_length[0] * s
    fy = focal_length[1] * s
    cx = (W / 2.0) - (principal_point[0] * s)
    cy = (H / 2.0) - (principal_point[1] * s)
    
    intri = np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ], dtype=np.float32)
    
    return extri.tolist(), intri.tolist()

def generate_annotations(dataset_root, category_name="human_body", test_split_ratio=0.1):
    annotations_dir = os.path.join(dataset_root, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Find all frame_annotations.jgz files
    anno_files = glob.glob(os.path.join(dataset_root, "*", "frame_annotations.jgz"))
    if not anno_files:
        print(f"No frame_annotations.jgz found in {dataset_root}")
        return
        
    train_data = {}
    test_data = {}
    
    for anno_file in anno_files:
        print(f"Processing {anno_file}...")
        base_dir = os.path.dirname(anno_file)
        
        with gzip.open(anno_file, "rt") as f:
            frames = json.load(f)
            
        for frame in frames:
            seq_name = frame["sequence_name"]
            img_rel_path = frame["image"]["path"]  # usually "<seq>/images/frame.jpg"
            # Some datasets have category folder implicitly
            # In our case, image_path mapping for VGGT: 
            # img_rel_path already contains the folder, e.g., human_body_00/sequence_001/images/...
            filepath = img_rel_path.replace("\\", "/")
            
            full_img_path = os.path.join(dataset_root, filepath)
            if not os.path.exists(full_img_path):
                continue
                
            try:
                with Image.open(full_img_path) as img:
                    W, H = img.size
            except Exception as e:
                print(f"Failed to read image {full_img_path}: {e}")
                continue
                
            vp = frame["viewpoint"]
            extri, intri = convert_pt3d_to_opencv(
                vp["R"], vp["T"], 
                vp["focal_length"], vp["principal_point"], 
                (W, H)
            )
            
            frame_dict = {
                "filepath": filepath,
                "extri": extri,
                "intri": intri
            }
            
            import hashlib
            # Consistent hash-based train/test split
            seq_hash = int(hashlib.md5(seq_name.encode()).hexdigest(), 16) % 100
            
            if seq_hash < (test_split_ratio * 100):
                if seq_name not in test_data:
                    test_data[seq_name] = []
                test_data[seq_name].append(frame_dict)
            else:
                if seq_name not in train_data:
                    train_data[seq_name] = []
                train_data[seq_name].append(frame_dict)

    train_out = os.path.join(annotations_dir, f"{category_name}_train.jgz")
    test_out = os.path.join(annotations_dir, f"{category_name}_test.jgz")
    
    with gzip.open(train_out, "wt") as f:
        json.dump(train_data, f)
    with gzip.open(test_out, "wt") as f:
        json.dump(test_data, f)
        
    train_seqs = len(train_data)
    test_seqs = len(test_data)
    print(f"Saved annotations to {annotations_dir}")
    print(f"Train sequences: {train_seqs}")
    print(f"Test sequences: {test_seqs}")

if __name__ == "__main__":
    generate_annotations("datasets/human_body", category_name="human_body")
