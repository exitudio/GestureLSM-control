import torch

from utils import rotation_conversions as rc


def stitch_chunk_latents(chunk_latents, pre_frames):
    """Convert [chunks, latent_len, dim] overlapping chunks to one sequence."""
    if chunk_latents.shape[0] == 1:
        return chunk_latents[:1]
    parts = [chunk_latents[:1]]
    parts.extend(
        chunk_latents[i:i + 1, pre_frames:]
        for i in range(1, chunk_latents.shape[0])
    )
    return torch.cat(parts, dim=1)


def lower_latents_to_trans(lower_latents, vq_model_lower, latent_scale, trans_mean, trans_std):
    rec_lower = vq_model_lower.decode_continuous(lower_latents * latent_scale)
    rec_trans_v = rec_lower[..., -3:] * trans_std + trans_mean
    rec_trans = torch.cumsum(rec_trans_v, dim=-2)
    rec_trans[..., 1] = rec_trans_v[..., 1]
    return rec_trans


def decode_latents(
        latents,
        vq_model_upper,
        vq_model_hands,
        vq_model_lower,
        latent_scale,
        trans_mean,
        trans_std,
        mean_upper,
        std_upper,
        mean_hands,
        std_hands,
        mean_lower,
        std_lower,
        pose_norm=True,
        use_trans=True,
        freeze_root=False,
        trans_offset=None):
    rec_up = latents[..., :128] * latent_scale
    rec_hn = latents[..., 128:256] * latent_scale
    rec_lo = latents[..., 256:] * latent_scale

    rec_upper = vq_model_upper.decode_continuous(rec_up)
    rec_hands = vq_model_hands.decode_continuous(rec_hn)
    rec_lower = vq_model_lower.decode_continuous(rec_lo)

    if use_trans:
        rec_trans_v = rec_lower[..., -3:] * trans_std + trans_mean
        rec_trans = torch.cumsum(rec_trans_v, dim=-2)
        rec_trans[..., 1] = rec_trans_v[..., 1]
        if trans_offset is not None:
            rec_trans = rec_trans + trans_offset.to(
                device=rec_trans.device,
                dtype=rec_trans.dtype,
            ).view(1, 1, 3)
        rec_lower = rec_lower[..., :-3]
    else:
        rec_trans = torch.zeros(
            (latents.shape[0], latents.shape[1], 3),
            device=latents.device,
            dtype=latents.dtype,
        )

    if freeze_root:
        rec_trans = rec_trans.detach()

    if pose_norm:
        rec_upper = rec_upper * std_upper + mean_upper
        rec_hands = rec_hands * std_hands + mean_hands
        rec_lower = rec_lower * std_lower + mean_lower

    return rec_upper, rec_hands, rec_lower, rec_trans


def latents_to_joints(
        latents,
        betas,
        smplx,
        inverse_selection_fn,
        joint_mask_upper,
        joint_mask_lower,
        joint_mask_hands,
        vq_model_upper,
        vq_model_hands,
        vq_model_lower,
        latent_scale,
        trans_mean,
        trans_std,
        mean_upper,
        std_upper,
        mean_hands,
        std_hands,
        mean_lower,
        std_lower,
        pose_norm=True,
        use_trans=True,
        freeze_root=True,
        trans_offset=None):
    rec_upper, rec_hands, rec_lower, rec_trans = decode_latents(
        latents,
        vq_model_upper,
        vq_model_hands,
        vq_model_lower,
        latent_scale,
        trans_mean,
        trans_std,
        mean_upper,
        std_upper,
        mean_hands,
        std_hands,
        mean_lower,
        std_lower,
        pose_norm=pose_norm,
        use_trans=use_trans,
        freeze_root=freeze_root,
        trans_offset=trans_offset,
    )

    bs_l, n_l = rec_lower.shape[:2]
    ru = rc.matrix_to_axis_angle(
        rc.rotation_6d_to_matrix(rec_upper.reshape(bs_l, n_l, 13, 6))
    ).reshape(bs_l * n_l, 13 * 3)
    rh = rc.matrix_to_axis_angle(
        rc.rotation_6d_to_matrix(rec_hands.reshape(bs_l, n_l, 30, 6))
    ).reshape(bs_l * n_l, 30 * 3)
    rl = rc.matrix_to_axis_angle(
        rc.rotation_6d_to_matrix(rec_lower[..., :54].reshape(bs_l, n_l, 9, 6))
    ).reshape(bs_l * n_l, 9 * 3)

    total = bs_l * n_l
    rec_pose = (
        inverse_selection_fn(ru, joint_mask_upper, total)
        + inverse_selection_fn(rl, joint_mask_lower, total)
        + inverse_selection_fn(rh, joint_mask_hands, total)
    )
    aa = rec_pose.reshape(total, 55, 3)
    body_betas = (
        betas[:, None, :]
        .expand(bs_l, n_l, -1)
        .reshape(total, -1)
        .to(latents.device)
        .float()
    )
    out = smplx(
        betas=body_betas,
        global_orient=aa[:, 0],
        body_pose=aa[:, 1:22].reshape(total, 63),
        jaw_pose=aa[:, 22],
        leye_pose=aa[:, 23],
        reye_pose=aa[:, 24],
        left_hand_pose=aa[:, 25:40].reshape(total, 45),
        right_hand_pose=aa[:, 40:55].reshape(total, 45),
        transl=rec_trans.reshape(total, 3).float(),
        expression=torch.zeros(total, 100, device=latents.device),
        return_verts=False,
    )
    return out.joints[:, :55].reshape(bs_l, n_l, 55, 3)


def latent_tensor_to_joints(latent, *args, **kwargs):
    sample = latent.squeeze(2).permute(0, 2, 1)
    return latents_to_joints(sample, *args, **kwargs)


def latent_batch_to_window_joints(latent, round_l, pose_length, pre_frames, *args, **kwargs):
    chunk_latents = latent.squeeze(2).permute(0, 2, 1)
    full_latents = stitch_chunk_latents(chunk_latents, pre_frames)
    full_joints = latents_to_joints(full_latents, *args, **kwargs)
    windows = []
    for i in range(chunk_latents.shape[0]):
        start = i * round_l
        end = start + pose_length
        windows.append(full_joints[:, start:end])
    return torch.cat(windows, dim=0)


def update_dynamic_seeds(x0, first_seed, pre_frames):
    chunk_latents = x0.squeeze(2).permute(0, 2, 1)
    seeds = [first_seed]
    seeds.extend(
        chunk_latents[i - 1:i, -pre_frames:]
        for i in range(1, chunk_latents.shape[0])
    )
    return torch.cat(seeds, dim=0)
