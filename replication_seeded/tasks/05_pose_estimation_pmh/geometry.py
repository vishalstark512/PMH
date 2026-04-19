"""
Geometric consistency for 3D human pose (17 joints, Human3.6M-style).
Bone lengths, joint angles, symmetry, and violation checks for PMH geometric loss and eval.
"""
import torch

# Human3.6M 17 joints: 0=Pelvis, 1=RHip, 2=RKnee, 3=RAnkle, 4=LHip, 5=LKnee, 6=LAnkle,
# 7=Spine1, 8=Neck, 9=Head, 10=Site, 11=LShoulder, 12=LElbow, 13=LWrist, 14=RShoulder, 15=RElbow, 16=RWrist
NUM_JOINTS = 17

# (parent, child) bone edges
SKELETON_EDGES = [
    (0, 7), (0, 1), (0, 4),           # pelvis -> spine, hips
    (1, 2), (2, 3),                   # R leg
    (4, 5), (5, 6),                   # L leg
    (7, 8), (8, 9),                   # spine -> neck -> head
    (8, 11), (11, 12), (12, 13),      # L arm
    (8, 14), (14, 15), (15, 16),      # R arm
]


def pose_to_bone_lengths(pose):
    """pose: (..., 17, 3). Returns (..., num_bones)."""
    edges = torch.tensor(SKELETON_EDGES, device=pose.device, dtype=torch.long)
    p = pose[..., edges[:, 0], :]  # (..., E, 3)
    c = pose[..., edges[:, 1], :]   # (..., E, 3)
    return (p - c).norm(dim=-1)


def pose_to_bone_vectors(pose):
    """pose: (..., 17, 3). Returns (..., num_bones, 3) unit vectors along bones."""
    edges = torch.tensor(SKELETON_EDGES, device=pose.device, dtype=torch.long)
    p = pose[..., edges[:, 0], :]
    c = pose[..., edges[:, 1], :]
    v = c - p
    length = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return v / length


def joint_angles_at_elbows_knees(pose):
    """Approximate joint angles at elbows and knees via adjacent bone vectors. pose: (..., 17, 3). Returns (..., 4): LElbow, RElbow, LKnee, RKnee (cos of angle)."""
    vecs = pose_to_bone_vectors(pose)  # (..., E, 3)
    # Indices in vecs: 10=LUpperArm, 11=LForearm, 12=RUpperArm, 13=RForearm, 3=LThigh, 4=LShin, 1=RThigh, 2=RShin
    # SKELETON_EDGES order: 0:(0,7), 1:(0,1), 2:(0,4), 3:(1,2), 4:(2,3), 5:(4,5), 6:(5,6), 7:(7,8), 8:(8,9), 9:(8,11), 10:(11,12), 11:(12,13), 12:(8,14), 13:(14,15), 14:(15,16)
    # Elbow L: bones 10 (11->12) and 11 (12->13) -> dot of 10 and 11
    # Elbow R: bones 13 and 14
    # Knee L: bones 5 (4->5) and 6 (5->6) -> dot of 5 and 6
    # Knee R: bones 3 and 4
    L_elbow = (vecs[..., 10, :] * vecs[..., 11, :]).sum(dim=-1)
    R_elbow = (vecs[..., 13, :] * vecs[..., 14, :]).sum(dim=-1)
    L_knee = (vecs[..., 5, :] * vecs[..., 6, :]).sum(dim=-1)
    R_knee = (vecs[..., 3, :] * vecs[..., 4, :]).sum(dim=-1)
    return torch.stack([L_elbow, R_elbow, L_knee, R_knee], dim=-1)


def symmetry_left_right(pose):
    """Pose (..., 17, 3). Return (..., 4): [L-R] for shoulder, elbow, hip, knee (x flipped for right)."""
    # Left: 11,12,13 (LShoulder, LElbow, LWrist), 4,5,6 (LHip, LKnee, LAnkle)
    # Right: 14,15,16, 1,2,3
    left_j = [11, 12, 4, 5]
    right_j = [14, 15, 1, 2]
    out = []
    for l, r in zip(left_j, right_j):
        L = pose[..., l, :]
        R = pose[..., r, :]
        R_flip = R.clone()
        R_flip[..., 0] = -R_flip[..., 0]  # flip X for symmetry
        out.append((L - R_flip).norm(dim=-1))
    return torch.stack(out, dim=-1)


def geometric_consistency_loss(pose_clean, pose_occluded):
    """Loss encouraging same geometric structure. pose_clean, pose_occluded: (B, 17, 3)."""
    bones_c = pose_to_bone_lengths(pose_clean)
    bones_o = pose_to_bone_lengths(pose_occluded)
    loss_bones = (bones_c - bones_o).pow(2).mean()
    angles_c = joint_angles_at_elbows_knees(pose_clean)
    angles_o = joint_angles_at_elbows_knees(pose_occluded)
    loss_angles = (angles_c - angles_o).pow(2).mean()
    sym_c = symmetry_left_right(pose_clean)
    sym_o = symmetry_left_right(pose_occluded)
    loss_sym = (sym_c - sym_o).pow(2).mean()
    return loss_bones + loss_angles + loss_sym


def geometric_violation_rate(pose_pred, pose_gt, bone_length_tol=0.5, angle_tol=0.5):
    """
    Fraction of samples where predicted pose has impossible geometry vs GT.
    pose_pred, pose_gt: (N, 17, 3).
    bone_length_tol: max relative error in bone length (0.5 = 50%) to count as violation.
    """
    bones_pred = pose_to_bone_lengths(pose_pred)   # (N, E)
    bones_gt = pose_to_bone_lengths(pose_gt)       # (N, E)
    rel_err = (bones_pred - bones_gt).abs() / (bones_gt.abs() + 1e-6)
    bone_violation = (rel_err > bone_length_tol).any(dim=-1).float()
    angles_pred = joint_angles_at_elbows_knees(pose_pred)
    angles_gt = joint_angles_at_elbows_knees(pose_gt)
    angle_err = (angles_pred - angles_gt).abs()
    angle_violation = (angle_err > angle_tol).any(dim=-1).float()
    return (bone_violation + angle_violation).clamp(max=1).mean().item()


def mpjpe(pred, gt):
    """Mean per-joint position error (L2). pred, gt: (N, 17, 3). Returns scalar in same units as input."""
    return (pred - gt).norm(dim=-1).mean().item()


def pampjpe(pred, gt):
    """Procrustes-aligned MPJPE. pred, gt: (N, 17, 3)."""
    pred_ = pred.clone()
    gt_ = gt.clone()
    # Center
    pred_ = pred_ - pred_.mean(dim=(1, 2), keepdim=True)
    gt_ = gt_ - gt_.mean(dim=(1, 2), keepdim=True)
    # Scale to unit norm
    s_pred = (pred_.pow(2).sum(dim=(1, 2), keepdim=True) + 1e-8).sqrt()
    s_gt = (gt_.pow(2).sum(dim=(1, 2), keepdim=True) + 1e-8).sqrt()
    pred_ = pred_ / s_pred
    gt_ = gt_ / s_gt
    # Per-sample rotation: H = pred^T gt, SVD, R = V U^T
    errors = []
    for i in range(pred_.size(0)):
        P = pred_[i].reshape(17, 3)
        G = gt_[i].reshape(17, 3)
        H = P.t() @ G
        U, _, Vh = torch.linalg.svd(H)
        R = Vh.t() @ U.t()
        if torch.det(R) < 0:
            Vh = Vh.clone()
            Vh[-1] *= -1
            R = Vh.t() @ U.t()
        P_aligned = (P @ R.t())
        errors.append((P_aligned - G).norm(dim=-1).mean())
    return torch.stack(errors).mean().item()
