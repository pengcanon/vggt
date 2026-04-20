# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np


from vggt.dependency.distortion import apply_distortion, iterative_undistortion, single_undistortion


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def unproject_depth_with_gt_cameras(
    depth_map,
    pred_extrinsics,
    gt_extrinsics_3x4: np.ndarray,
    gt_intrinsics_native: np.ndarray,
    original_image_sizes,
):
    """
    Unproject predicted depth maps into 3D using ground-truth camera parameters.

    The predicted depth maps live in the model's internal normalised scale.  A
    single scale factor is estimated by comparing the camera-baseline implied by
    the predicted cameras against the baseline implied by the GT cameras, and that
    factor is applied to the depth before unprojection.

    GT intrinsics are rescaled from their native (original) pixel resolution to
    the 518-px resolution that ``load_and_preprocess_images(mode='crop')``
    produces, because that is the pixel grid the depth maps are aligned to.

    GT extrinsics are normalised so that camera-0 is at the world origin
    (identity pose).  All other cameras are expressed relative to camera-0.

    Args:
        depth_map: Predicted depth maps, shape (S, H, W) or (S, H, W, 1).
                   Accepts both torch.Tensor and np.ndarray.
        pred_extrinsics: Predicted extrinsic matrices, shape (S, 3, 4) in
                         OpenCV convention (camera from world).
                         Accepts both torch.Tensor and np.ndarray.
        gt_extrinsics_3x4 (np.ndarray): Ground-truth extrinsics already
                         converted to **OpenCV convention** (x-right, y-down,
                         z-forward), shape (S, 3, 4).  Raw PyTorch3D
                         frame_annotations must be converted first via
                         ``convert_pt3d_to_opencv`` before being passed here.
        gt_intrinsics_native (np.ndarray): Ground-truth intrinsics in the
                         **original** image pixel space, shape (S, 3, 3).
                         Must have already been denormalised from PyTorch3D's
                         normalised focal/principal-point representation.
        original_image_sizes: Sequence of (W, H) tuples, one per frame, giving
                         the pixel dimensions of the original (un-resized) images.

    Returns:
        tuple:
            - point_map (np.ndarray): 3D world points shape (S, H, W, 3) expressed
              in the coordinate frame of GT camera-0.
            - depth_scale (float): Scale factor applied to depth
              (VGGT normalised units → GT metric units).
    """
    # ------------------------------------------------------------------ #
    # Convert inputs to numpy                                              #
    # ------------------------------------------------------------------ #
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().float().numpy()
    if isinstance(pred_extrinsics, torch.Tensor):
        pred_extrinsics = pred_extrinsics.cpu().float().numpy()

    depth_map = np.asarray(depth_map, dtype=np.float32)
    pred_extrinsics = np.asarray(pred_extrinsics, dtype=np.float32)
    gt_extrinsics_3x4 = np.asarray(gt_extrinsics_3x4, dtype=np.float32)
    gt_intrinsics_native = np.asarray(gt_intrinsics_native, dtype=np.float32)

    S = depth_map.shape[0]
    assert gt_extrinsics_3x4.shape == (S, 3, 4), f"gt_extrinsics_3x4 shape mismatch: {gt_extrinsics_3x4.shape}"
    assert gt_intrinsics_native.shape == (S, 3, 3), f"gt_intrinsics_native shape mismatch: {gt_intrinsics_native.shape}"
    assert len(original_image_sizes) == S

    # ------------------------------------------------------------------ #
    # 1. Rescale GT intrinsics to 518-px model-input space                #
    #    Replicates load_and_preprocess_images(mode="crop"):               #
    #      - resize width to 518, maintain aspect ratio (÷14 rounding)    #
    #      - centre-crop height if new_height > 518                        #
    # ------------------------------------------------------------------ #
    TARGET = 518
    gt_intrinsics_518 = np.zeros_like(gt_intrinsics_native)
    for i, (W_orig, H_orig) in enumerate(original_image_sizes):
        H_resized = round(H_orig * (TARGET / W_orig) / 14) * 14
        sw = TARGET / W_orig          # width scale
        sh = H_resized / H_orig       # height scale
        crop_y = (H_resized - TARGET) // 2 if H_resized > TARGET else 0

        K = gt_intrinsics_native[i].copy()
        K[0, 0] *= sw          # fx
        K[1, 1] *= sh          # fy
        K[0, 2] *= sw          # cx
        K[1, 2]  = K[1, 2] * sh - crop_y  # cy: scale then subtract crop offset
        gt_intrinsics_518[i] = K

    # ------------------------------------------------------------------ #
    # 2. Normalise GT extrinsics so camera-0 is the world origin          #
    # ------------------------------------------------------------------ #
    bottom = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (S, 1, 1))  # (S,1,4)
    gt_ext_4x4 = np.concatenate([gt_extrinsics_3x4, bottom], axis=1)  # (S, 4, 4)

    # closed_form_inverse_se3 expects a batch (N, 4, 4)
    first_inv = closed_form_inverse_se3(gt_ext_4x4[0:1])   # (1, 4, 4)
    gt_ext_rel_4x4 = gt_ext_4x4 @ first_inv[0]             # (S, 4, 4); cam-0 → identity
    gt_ext_rel_3x4 = gt_ext_rel_4x4[:, :3, :]              # (S, 3, 4)

    # ------------------------------------------------------------------ #
    # 3. Estimate depth scale via camera-baseline ratio                   #
    #    Predicted camera centres: C_i = -R_i^T @ t_i                    #
    #    Baseline_pred = ||C_i - C_0||  (accounts for un-normalised cams) #
    #    Baseline_GT   = ||t_i_rel|| (GT cam-0 is at origin after step 2) #
    #                                                                      #
    #    Uses the *median* of per-pair scale ratios for robustness.        #
    #    A depth-scale error ε causes a world-space shift of               #
    #    ε·(p_world − C_i) per view, which is view-dependent when         #
    #    cameras sit at different elevations.  Using median reduces        #
    #    the impact of outlier camera-pair ratios.                         #
    # ------------------------------------------------------------------ #
    if S > 1:
        # Predicted camera centres
        R_pred = pred_extrinsics[:, :3, :3]   # (S, 3, 3)
        t_pred = pred_extrinsics[:, :3, 3]    # (S, 3)
        C_pred = -np.einsum("sij,sj->si", R_pred.transpose(0, 2, 1), t_pred)  # (S, 3)
        pred_baselines = np.linalg.norm(C_pred[1:] - C_pred[0:1], axis=-1)    # (S-1,)

        # GT camera centres relative to cam-0 (which sits at origin)
        # For an OpenCV extrinsic [R|t], the camera centre is -R^T t.
        # Since cam-0 is identity after normalisation, GT cam-0 centre = [0,0,0].
        gt_baselines = np.linalg.norm(gt_ext_rel_3x4[1:, :3, 3], axis=-1)     # (S-1,)

        # Per-pair scale ratios: gt_baseline_i / pred_baseline_i
        valid = pred_baselines > 1e-8
        if valid.any():
            per_pair_scales = gt_baselines[valid] / pred_baselines[valid]
            depth_scale = float(np.median(per_pair_scales))
        else:
            depth_scale = 1.0
    else:
        depth_scale = 1.0

    # ------------------------------------------------------------------ #
    # 4. Scale depth and unproject                                        #
    # ------------------------------------------------------------------ #
    scaled_depth = depth_map * depth_scale
    point_map = unproject_depth_map_to_point_map(scaled_depth, gt_ext_rel_3x4, gt_intrinsics_518)

    return point_map, depth_scale


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


# TODO: this code can be further cleaned up


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape BxSxHxWx3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape BxSx3x4.
    Returns:
    """
    # TODO: merge this into project_world_points_to_cam
    
    # device = world_points.device
    # with torch.autocast(device_type=device.type, enabled=False):
    ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
    world_points_h = torch.cat([world_points, ones], dim=-1)  # shape: (B, S, H, W, 4)

    # extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # Now perform the matrix multiplication
    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1)
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points



def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics=None,
    distortion_params=None,
    default=0,
    only_points_cam=False,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape Px3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
        distortion_params (torch.Tensor): Extra parameters of shape BxN, which is used for radial distortion.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    # with torch.autocast(device_type=device.type, dtype=torch.double):
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # Number of points
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        if only_points_cam:
            return None, cam_points

        # Step 2: Apply intrinsic parameters and (optional) distortion
        image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

        return image_points, cam_points



def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
    """
    Applies intrinsic parameters and optional distortion to the given 3D points.

    Args:
        cam_intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
        cam_points (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
        distortion_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
        default (float, optional): Default value to replace NaNs in the output.

    Returns:
        pixel_coords (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
    """

    # Normalized device coordinates (NDC)
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # Apply distortion if distortion_params are provided
    if distortion_params is not None:
        x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
        distorted_xy = torch.stack([x_distorted, y_distorted], dim=1)
    else:
        distorted_xy = ndc_xy

    # Prepare cam_points for batch matrix multiplication
    cam_coords_homo = torch.cat(
        (distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1
    )  # Bx3xN
    # Apply intrinsic parameters using batch matrix multiplication
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # Extract x and y coordinates
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # Replace NaNs with default value
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2




def cam_from_img(pred_tracks, intrinsics, extra_params=None):
    """
    Normalize predicted tracks based on camera intrinsics.
    Args:
    intrinsics (torch.Tensor): The camera intrinsics tensor of shape [batch_size, 3, 3].
    pred_tracks (torch.Tensor): The predicted tracks tensor of shape [batch_size, num_tracks, 2].
    extra_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    Returns:
    torch.Tensor: Normalized tracks tensor.
    """

    # We don't want to do intrinsics_inv = torch.inverse(intrinsics) here
    # otherwise we can use something like
    #     tracks_normalized_homo = torch.bmm(pred_tracks_homo, intrinsics_inv.transpose(1, 2))

    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    tracks_normalized = (pred_tracks - principal_point) / focal_length

    if extra_params is not None:
        # Apply iterative undistortion
        try:
            tracks_normalized = iterative_undistortion(
                extra_params, tracks_normalized
            )
        except:
            tracks_normalized = single_undistortion(
                extra_params, tracks_normalized
            )

    return tracks_normalized