import os
import glob
import json
import gzip
import numpy as np
from PIL import Image

def get_opencv_matrices(vp):
    R_cv = np.array(vp["R"], dtype=np.float32)
    T_cv = np.array(vp["T"], dtype=np.float32)
    
    extri = np.hstack((R_cv, T_cv.reshape(3, 1)))
    
    fx, fy = vp["focal_length"]
    cx, cy = vp["principal_point"]
    
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
                
            vp = frame["viewpoint"]
            extri, intri = get_opencv_matrices(vp)
            
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
