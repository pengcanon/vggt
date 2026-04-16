import json, gzip, numpy as np

co3d_anno_path = "datasets/co3d/dataset/multi_view/apple/frame_annotations.jgz"
vggt_anno_file = "datasets/co3d/annotations/apple_test.jgz"

try:
    with gzip.open(co3d_anno_path, "rt") as f:
        pt3d_data = json.load(f)
    print("Found pt3d frames:", len(pt3d_data))
    for d in pt3d_data:
        if d["sequence_name"] == "110_13051_23361" and ("000001" in d["image"]["path"] or "frame000001" in d["image"]["path"]):
            r = np.array(d["viewpoint"]["R"])
            t = np.array(d["viewpoint"]["T"])
            print("PT3D R:\n", r)
            print("PT3D T:\n", t)
            print("PT3D focal:", d["viewpoint"]["focal_length"])
            print("PT3D prin:", d["viewpoint"]["principal_point"])
            break
            
    with gzip.open(vggt_anno_file, "rt") as f:
        data = json.load(f)
        vggt_anno = data["110_13051_23361"][0]
        print("VGGT extri:\n", np.array(vggt_anno["extri"]))
        print("VGGT intri:\n", np.array(vggt_anno["intri"]))

except Exception as e:
    print(e)
