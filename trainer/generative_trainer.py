import os
import pprint
import shutil
import random
import sys
import time
import warnings
from typing import Dict
from datetime import datetime

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from dataloaders import data_tools
from dataloaders.data_tools import joints_list
from loguru import logger
from models.vq.model import RVQVAE
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
from utils import (
    data_transfer,
    guidance as guidance_utils,
    logger_tools,
    metric,
    other_tools,
    other_tools_hf,
    rotation_conversions as rc,
)
from utils.joints import hands_body_mask, lower_body_mask, upper_body_mask


def convert_15d_to_6d(motion):
    """
    Convert 15D motion to 6D motion, the current motion is 15D, but the eval model is 6D
    """
    bs = motion.shape[0]
    motion_6d = motion.reshape(bs, -1, 55, 15)[:, :, :, 6:12]
    motion_6d = motion_6d.reshape(bs, -1, 55 * 6)
    return motion_6d


SKATING_METRIC_MAP = (
    ("skating_ratio", "ratio_050"),
    ("skating_ratio_010", "ratio_010"),
    ("skating_ratio_020", "ratio_020"),
    ("skating_speed", "speed"),
    ("skating_distance", "distance"),
)
SKATING_METRIC_NAMES = tuple(name for name, _ in SKATING_METRIC_MAP)
GT_SKATING_METRIC_NAMES = tuple(f"gt_{name}" for name in SKATING_METRIC_NAMES)

CONTROL_JOINT_GROUPS = {
    "head": (15,),
    "left_elbow": (18,),
    "right_elbow": (19,),
    "left_wrist": (20,),
    "right_wrist": (21,),
    "all": (20, 21, 18, 19),
}
CONTROL_JOINT_NAMES = {
    15: "head",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}
CONTROL_DENSITIES = (1, 2, 5)
CONTROL_ERROR_THRESHOLDS_CM = (5, 10, 20, 50)
CONTROL_TRAJ_METRIC_NAMES = tuple(
    f"traj_err_{threshold_cm}cm" for threshold_cm in CONTROL_ERROR_THRESHOLDS_CM
)
CONTROL_LOC_METRIC_NAMES = tuple(
    f"loc_err_{threshold_cm}cm" for threshold_cm in CONTROL_ERROR_THRESHOLDS_CM
)
CONTROL_METRIC_NAMES = (
    "fgd",
    "gt_fgd",
    "align",
    "gt_align",
    "l1div",
    "gt_l1div",
    "foot_skating_ratio",
    *CONTROL_TRAJ_METRIC_NAMES,
    *CONTROL_LOC_METRIC_NAMES,
    "avg_err_cm",
)


def _control_cfg_value(cfg, name, default):
    control_cfg = getattr(cfg, "control_eval", None)
    if control_cfg is None or not hasattr(control_cfg, name):
        return default
    return getattr(control_cfg, name)


def _parse_control_joint_groups(groups):
    if not isinstance(groups, str):
        return groups

    text = groups.strip()
    if not text:
        return []

    if text.startswith("[[") and text.endswith("]]" ):
        inner = text[2:-2]
        return [
            [name.strip().strip("'\"") for name in group.split(",") if name.strip()]
            for group in inner.split("],[")
        ]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        return [[name.strip().strip("'\"") for name in inner.split(",") if name.strip()]]
    return [[text.strip("'\"")]]


def _normalize_control_joint_group(group):
    if isinstance(group, str):
        group = group.strip().strip("'\"")
        if group not in CONTROL_JOINT_GROUPS:
            raise ValueError(f"Unknown control joint group: {group}")
        return group, CONTROL_JOINT_GROUPS[group]

    names = [str(name).strip().strip("'\"") for name in group]
    if names in (["left_wrist", "right_wrist", "left_elbow", "right_elbow"], ["head", "left_wrist", "right_wrist"]):
        if names == ["head", "left_wrist", "right_wrist"]:
            return "head_left_wrist_right_wrist", (15, 20, 21)
        return "all", CONTROL_JOINT_GROUPS["all"]
    joints = []
    for name in names:
        if name not in CONTROL_JOINT_GROUPS or len(CONTROL_JOINT_GROUPS[name]) != 1:
            raise ValueError(f"Unknown control joint name: {name}")
        joints.append(CONTROL_JOINT_GROUPS[name][0])
    return "_".join(names), tuple(joints)


def _control_eval_settings_from_cfg(cfg):
    groups = _parse_control_joint_groups(_control_cfg_value(
        cfg,
        "joints",
        (
            ("left_wrist",),
            ("right_wrist",),
            ("left_elbow",),
            ("right_elbow",),
            ("left_wrist", "right_wrist", "left_elbow", "right_elbow"),
        ),
    ))
    densities = _control_cfg_value(cfg, "densities", CONTROL_DENSITIES)
    space = str(_control_cfg_value(cfg, "space", "absolute")).strip().lower()
    if space not in ("absolute", "relative"):
        raise ValueError(f"Unknown control_eval.space: {space}")
    normalized_groups = [_normalize_control_joint_group(group) for group in groups]
    single_groups = [item for item in normalized_groups if len(item[1]) == 1]
    multi_groups = [item for item in normalized_groups if len(item[1]) != 1]

    settings = []
    for group_list in (single_groups, multi_groups):
        for density in densities:
            density = int(density)
            density_name = "chunk_end" if density == -1 else str(density)
            for group_name, joints in group_list:
                settings.append({
                    "name": f"{group_name}/density_{density_name}",
                    "joint_group": group_name,
                    "joints": joints,
                    "density": density,
                    "space": space,
                })
    return settings


def _control_eval_metric_names(settings):
    return [f"control/aggregate/{metric_name}" for metric_name in CONTROL_METRIC_NAMES]


class CustomTrainer(BaseTrainer):
    """
    Generative Trainer to support various generative models
    """

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        self.joints = 55

        self.ori_joint_list = joints_list["beat_smplx_joints"]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1

        self.control_eval_settings = _control_eval_settings_from_cfg(cfg)
        self.control_eval_metric_names = _control_eval_metric_names(
            self.control_eval_settings
        )
        self.tracker = other_tools.EpochTracker(
            [
                "fgd", "gt_fgd", "bc", "gt_bc", "l1div", "gt_l1div",
                *SKATING_METRIC_NAMES, *GT_SKATING_METRIC_NAMES,
                *self.control_eval_metric_names,
                "predict_x0_loss", "test_clip_fgd",
            ],
            [
                True, False, True, True, True, True,
                *([False] * len(SKATING_METRIC_NAMES)),
                *([False] * len(GT_SKATING_METRIC_NAMES)),
                *([False] * len(self.control_eval_metric_names)),
                True, True,
            ],
        )

        ##### Model #####

        model_module = __import__(
            f"models.{cfg.model.model_name}", fromlist=["something"]
        )

        if self.cfg.ddp:
            self.model = getattr(model_module, cfg.model.g_name)(cfg).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model, process_group
            )
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = torch.nn.DataParallel(
                getattr(model_module, cfg.model.g_name)(cfg), self.cfg.gpus
            ).cuda()

        if self.args.mode == "train":
            if self.rank == 0:
                logger.info(self.model)
                logger.info(f"init {self.cfg.model.g_name} success")
                wandb.watch(self.model)

        ##### Optimizer and Scheduler #####
        self.opt = create_optimizer(self.cfg.solver, self.model)
        self.opt_s = create_scheduler(self.cfg.solver, self.opt)

        ##### VQ-VAE models #####
        """Initialize and load VQ-VAE models for different body parts."""
        # Body part VQ models
        self.vq_models = self._create_body_vq_models()

        # Set all VQ models to eval mode
        for model in self.vq_models.values():
            model.eval().to(self.rank)

        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = (
            self.vq_models.values()
        )

        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction="mean").to(self.rank)

        ##### Normalization #####
        self.mean = np.load("./mean_std/beatx_2_330_mean.npy")
        self.std = np.load("./mean_std/beatx_2_330_std.npy")

        # Extract body part specific normalizations
        for part in ["upper", "hands", "lower"]:
            mask = globals()[f"{part}_body_mask"]
            setattr(self, f"mean_{part}", torch.from_numpy(self.mean[mask]).cuda())
            setattr(self, f"std_{part}", torch.from_numpy(self.std[mask]).cuda())

        self.trans_mean = torch.from_numpy(
            np.load("./mean_std/beatx_2_trans_mean.npy")
        ).cuda()
        self.trans_std = torch.from_numpy(
            np.load("./mean_std/beatx_2_trans_std.npy")
        ).cuda()

        if self.args.checkpoint:
            try:
                ckpt_state_dict = torch.load(self.args.checkpoint, weights_only=False)[
                    "model_state_dict"
                ]
            except:
                ckpt_state_dict = torch.load(self.args.checkpoint, weights_only=False)[
                    "model_state"
                ]
            # remove 'audioEncoder' from the state_dict due to legacy issues
            ckpt_state_dict = {
                k: v
                for k, v in ckpt_state_dict.items()
                if "modality_encoder.audio_encoder." not in k
            }
            self.model.load_state_dict(ckpt_state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {self.args.checkpoint}")

    def _create_body_vq_models(self) -> Dict[str, RVQVAE]:
        """Create VQ-VAE models for body parts."""
        vq_configs = {
            "upper": {"dim_pose": 78},
            "hands": {"dim_pose": 180},
            "lower": {"dim_pose": 57},
        }

        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config["dim_pose"], part)
            vq_models[part] = model

        return vq_models

    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        """Create a single RVQVAE model with specified configuration."""

        vq_args = self.args
        vq_args.num_quantizers = 6
        vq_args.shared_codebook = False
        vq_args.quantize_dropout_prob = 0.2
        vq_args.quantize_dropout_cutoff_index = 0
        vq_args.mu = 0.99
        vq_args.beta = 1.0
        model = RVQVAE(
            vq_args,
            input_width=dim_pose,
            nb_code=1024,
            code_dim=128,
            output_emb_width=128,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

        # Load pretrained weights
        checkpoint_path = getattr(self.cfg, f"vqvae_{body_part}_path")
        model.load_state_dict(torch.load(checkpoint_path)["net"])
        return model

    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def _load_data(self, dict_data):
        facial_rep = dict_data["facial"].to(self.rank)
        beta = dict_data["beta"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank)

        # process the pose data
        tar_pose = dict_data["pose"][:, :, :165].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        tar_pose_hands = tar_pose[:, :, 25 * 3 : 55 * 3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30 * 6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13 * 6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9 * 6)

        tar_pose_lower = tar_pose_leg

        tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
        tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
        tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower

        tar_trans_v = (tar_trans_v - self.trans_mean) / self.trans_std
        tar_pose_lower = torch.cat([tar_pose_lower, tar_trans_v], dim=-1)

        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        

        ## TODO: Whether the latent scale is needed here?
        # latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)
        latent_in = (
            torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2) / 5
        )

        word = dict_data.get("word", None)
        if word is not None:
            word = word.to(self.rank)

        # style feature is always None (without annotation, we never know what it is)
        style_feature = None

        audio_onset = None
        if self.cfg.data.onset_rep:
            audio_onset = dict_data["audio_onset"].to(self.rank)

        return {
            "audio_onset": audio_onset,
            "word": word,
            "latent_in": latent_in,
            "tar_id": tar_id,
            "facial_rep": facial_rep,
            "beta": beta,
            "tar_pose": tar_pose,
            "trans": tar_trans,
            "style_feature": style_feature,
        }

    def _g_training(self, loaded_data, mode="train", epoch=0):
        self.model.train()
        cond_ = {"y": {}}
        cond_["y"]["audio_onset"] = loaded_data["audio_onset"]
        cond_["y"]["word"] = loaded_data["word"]
        cond_["y"]["id"] = loaded_data["tar_id"]
        cond_["y"]["seed"] = loaded_data["latent_in"][:, : self.cfg.pre_frames]
        cond_["y"]["style_feature"] = loaded_data["style_feature"]
        x0 = loaded_data["latent_in"]
        x0 = x0.permute(0, 2, 1).unsqueeze(2)

        g_loss_final = self.model.module.train_forward(cond_, x0)["loss"]

        self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())

        if mode == "train":
            return g_loss_final

    def _control_eval_enabled(self):
        control_cfg = getattr(self.cfg, "control_eval", None)
        return bool(control_cfg is not None and getattr(control_cfg, "enabled", False))

    def _control_eval_threshold_m(self):
        return float(_control_cfg_value(self.cfg, "threshold_m", 0.05))

    def _axis_angle_to_smplx_joints(self, pose_aa, trans, beta, exps=None):
        bs, n = pose_aa.shape[:2]
        pose_flat = pose_aa.reshape(bs * n, self.joints * 3)
        beta_flat = beta.reshape(bs * n, -1)
        trans_flat = trans.reshape(bs * n, 3)
        if exps is None:
            exps_flat = torch.zeros(bs * n, 100, device=pose_aa.device, dtype=pose_aa.dtype)
        else:
            exps_flat = exps.reshape(bs * n, 100)
        out = self.smplx(
            betas=beta_flat,
            transl=trans_flat,
            expression=exps_flat,
            jaw_pose=pose_flat[:, 66:69],
            global_orient=pose_flat[:, :3],
            body_pose=pose_flat[:, 3 : 21 * 3 + 3],
            left_hand_pose=pose_flat[:, 25 * 3 : 40 * 3],
            right_hand_pose=pose_flat[:, 40 * 3 : 55 * 3],
            return_joints=True,
            leye_pose=pose_flat[:, 69:72],
            reye_pose=pose_flat[:, 72:75],
        )
        return out["joints"].reshape(bs, n, 127, 3)

    def _lower_latents_to_trans(self, lower_latents):
        return guidance_utils.lower_latents_to_trans(
            lower_latents,
            self.vq_model_lower,
            self.cfg.vqvae_latent_scale,
            self.trans_mean,
            self.trans_std,
        )

    def _latent_chunk_to_joints(self, latent, betas, freeze_root=True, trans_offset=None):
        return guidance_utils.latent_tensor_to_joints(
            latent,
            betas,
            self.smplx,
            self.inverse_selection_tensor,
            self.joint_mask_upper,
            self.joint_mask_lower,
            self.joint_mask_hands,
            self.vq_model_upper,
            self.vq_model_hands,
            self.vq_model_lower,
            self.cfg.vqvae_latent_scale,
            self.trans_mean,
            self.trans_std,
            self.mean_upper,
            self.std_upper,
            self.mean_hands,
            self.std_hands,
            self.mean_lower,
            self.std_lower,
            pose_norm=getattr(self.cfg.data, "pose_norm", True),
            use_trans=True,
            freeze_root=freeze_root,
            trans_offset=trans_offset,
        )

    @staticmethod
    def _to_control_space(joints, space, root_rot=None):
        if space != "relative":
            return joints
        if root_rot is None:
            raise ValueError("control space 'relative' requires root rotations")
        rel = joints - joints[..., 0:1, :]
        if torch.is_tensor(rel):
            return torch.matmul(
                root_rot.transpose(-1, -2).unsqueeze(-3),
                rel.unsqueeze(-1),
            ).squeeze(-1)
        return np.einsum("...ji,...kj->...ki", root_rot, rel)

    def _latent_batch_to_window_joints(self, latent, betas, freeze_root, round_l, control_space="absolute"):
        decoded = guidance_utils.latent_batch_to_window_joints(
            latent,
            round_l,
            self.cfg.data.pose_length,
            self.cfg.pre_frames,
            betas,
            self.smplx,
            self.inverse_selection_tensor,
            self.joint_mask_upper,
            self.joint_mask_lower,
            self.joint_mask_hands,
            self.vq_model_upper,
            self.vq_model_hands,
            self.vq_model_lower,
            self.cfg.vqvae_latent_scale,
            self.trans_mean,
            self.trans_std,
            self.mean_upper,
            self.std_upper,
            self.mean_hands,
            self.std_hands,
            self.mean_lower,
            self.std_lower,
            pose_norm=getattr(self.cfg.data, "pose_norm", True),
            use_trans=True,
            freeze_root=freeze_root,
            return_root_rot=(control_space == "relative"),
        )
        if control_space == "relative":
            joints, root_rot = decoded
        else:
            joints, root_rot = decoded, None
        return self._to_control_space(joints, control_space, root_rot)

    def _latent_delayed_window_joints(
            self,
            latent,
            betas,
            freeze_root,
            round_l,
            control_space="absolute",
            full_x0=None,
            chunk_indices=None):
        if full_x0 is None or chunk_indices is None:
            return self._latent_batch_to_window_joints(
                latent, betas, freeze_root, round_l, control_space
            )

        if torch.is_tensor(chunk_indices):
            indices = [int(i) for i in chunk_indices.detach().cpu().tolist()]
        else:
            indices = [int(i) for i in chunk_indices]

        full_context = full_x0.clone()
        full_context[indices] = latent
        all_windows = self._latent_batch_to_window_joints(
            full_context, betas, freeze_root, round_l, control_space
        )
        return all_windows[indices]

    def _update_dynamic_seeds(self, x0, first_seed):
        return guidance_utils.update_dynamic_seeds(x0, first_seed, self.cfg.pre_frames)

    def _build_control_from_target(self, target_joints, setting, rng, root_rot=None):
        control_space = setting.get("space", "absolute")
        target_joints = self._to_control_space(
            target_joints[:, :, :55],
            control_space,
            root_rot,
        )
        bs, n = target_joints.shape[:2]
        hint = torch.zeros_like(target_joints)
        mask = torch.zeros(bs, n, 55, device=target_joints.device, dtype=target_joints.dtype)
        density = int(setting["density"])
        seed_frames = self.cfg.pre_frames * self.cfg.vqvae_squeeze_scale
        round_l = self.cfg.data.pose_length - seed_frames
        if density == -1:
            frames = list(range(self.cfg.data.pose_length - 1, n, round_l))
            if not frames or frames[-1] != n - 1:
                frames.append(n - 1)
            frames_by_batch = [np.asarray(frames, dtype=np.int64) for _ in range(bs)]
        else:
            frames_by_batch = []
            for b in range(bs):
                selected = []
                for start in range(0, n, round_l):
                    chunk_start = start if start == 0 else start + seed_frames
                    chunk_end = min(start + self.cfg.data.pose_length, n)
                    if chunk_start >= chunk_end:
                        continue
                    eligible = np.arange(chunk_start, chunk_end, dtype=np.int64)
                    if density in (1, 2, 5):
                        frame_count = min(len(eligible), density)
                    else:
                        frame_count = min(len(eligible), int(len(eligible) * density / 100))
                    if frame_count > 0:
                        selected.extend(rng.choice(eligible, frame_count, replace=False).tolist())
                if not selected:
                    return None
                frames_by_batch.append(np.asarray(sorted(set(selected)), dtype=np.int64))
        for b, frames in enumerate(frames_by_batch):
            for joint in setting["joints"]:
                hint[b, frames, joint] = target_joints[b, frames, joint]
                mask[b, frames, joint] = 1.0
        if mask.sum() <= 0:
            return None
        return {"hint": hint, "mask": mask}

    def _g_test(self, loaded_data, control_setting=None, control_rng=None):
        self.model.eval()
        tar_beta = loaded_data["beta"]
        tar_pose = loaded_data["tar_pose"]
        tar_exps = loaded_data["facial_rep"]
        tar_trans = loaded_data["trans"]

        audio_onset = loaded_data["audio_onset"]
        in_word = loaded_data["word"]

        in_x0 = loaded_data["latent_in"]
        in_seed = loaded_data["latent_in"]

        bs, n, j = (
            loaded_data["tar_pose"].shape[0],
            loaded_data["tar_pose"].shape[1],
            self.joints,
        )

        remain = n % 8
        if remain != 0:

            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_exps = tar_exps[:, :-remain, :]
            in_x0 = in_x0[
                :, : in_x0.shape[1] - (remain // self.cfg.vqvae_squeeze_scale), :
            ]
            in_seed = in_seed[
                :, : in_x0.shape[1] - (remain // self.cfg.vqvae_squeeze_scale), :
            ]
            in_word = in_word[:, :-remain]
            n = n - remain

        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.cfg.vqvae_squeeze_scale
        pre_frames_scaled = self.cfg.pre_frames * vqvae_squeeze_scale
        roundt = (n - pre_frames_scaled) // (
            self.cfg.data.pose_length - pre_frames_scaled
        )
        remain = (n - pre_frames_scaled) % (
            self.cfg.data.pose_length - pre_frames_scaled
        )
        round_l = self.cfg.pose_length - pre_frames_scaled
        round_audio = int(round_l / 3 * 5)

        final_n = n - remain
        control_full = None
        if control_setting is not None:
            if self.cfg.model.g_name != "GestureDiffusion":
                raise ValueError("control_eval requires a GestureDiffusion model")
            with torch.no_grad():
                target_joints = self._axis_angle_to_smplx_joints(
                    tar_pose[:, :final_n],
                    tar_trans[:, :final_n],
                    tar_beta[:, :final_n],
                    tar_exps[:, :final_n],
                )[:, :, :55]
                target_root_rot = rc.axis_angle_to_matrix(
                    tar_pose[:, :final_n, :3].reshape(-1, 3)
                ).reshape(bs, final_n, 3, 3)
            control_full = self._build_control_from_target(
                target_joints, control_setting, control_rng, target_root_rot
            )
            if control_full is not None:
                control_full["space"] = control_setting.get("space", "absolute")

        if control_full is not None:
            if bs != 1:
                raise ValueError("Batched control_eval guidance currently requires test batch_size=1")
            audio_chunks, word_chunks, id_chunks, seed_chunks = [], [], [], []
            first_seed = in_seed[:, : self.cfg.pre_frames, :]
            for i in range(0, roundt):
                if audio_onset is not None:
                    audio_chunks.append(
                        audio_onset[
                            :,
                            i * (16000 // 30 * round_l) : (i + 1) * (16000 // 30 * round_l)
                            + 16000 // 30 * self.cfg.pre_frames * vqvae_squeeze_scale,
                        ]
                    )
                if in_word is not None:
                    word_chunks.append(
                        in_word[
                            :,
                            i * round_l : (i + 1) * round_l
                            + self.cfg.pre_frames * vqvae_squeeze_scale,
                        ]
                    )
                id_chunks.append(
                    loaded_data["tar_id"][
                        :, i * round_l : (i + 1) * round_l + self.cfg.pre_frames
                    ]
                )
                seed_chunks.append(torch.zeros_like(first_seed) if i > 0 else first_seed)

            hint_chunks, mask_chunks = [], []
            for i in range(0, roundt):
                start = i * round_l
                end = start + self.cfg.data.pose_length
                chunk_hint = control_full["hint"][:, start:end].clone()
                chunk_mask = control_full["mask"][:, start:end].clone()
                if i > 0:
                    chunk_mask[:, :pre_frames_scaled] = 0
                hint_chunks.append(chunk_hint)
                mask_chunks.append(chunk_mask)

            hint_batch = torch.cat(hint_chunks, dim=0)
            mask_batch = torch.cat(mask_chunks, dim=0)
            cond_ = {"y": {}}
            cond_["y"]["audio_onset"] = torch.cat(audio_chunks, dim=0) if audio_chunks else None
            cond_["y"]["word"] = torch.cat(word_chunks, dim=0) if word_chunks else None
            cond_["y"]["id"] = torch.cat(id_chunks, dim=0)
            cond_["y"]["seed"] = torch.cat(seed_chunks, dim=0)
            cond_["y"]["style_feature"] = torch.zeros(
                [roundt * bs, 512], device=self.mean_upper.device
            )
            if mask_batch.sum() > 0:
                cond_["y"]["control"] = {
                    "hint": hint_batch,
                    "mask": mask_batch,
                    "iters_early": int(_control_cfg_value(self.cfg, "iters_early", 5)),
                    "iters_late": int(_control_cfg_value(self.cfg, "iters_late", 30)),
                    "late_start": int(_control_cfg_value(self.cfg, "late_start", 300)),
                    "post_iters": int(_control_cfg_value(self.cfg, "post_iters", 0)),
                    "scale": float(_control_cfg_value(self.cfg, "scale", 20.0)),
                    "weight": float(_control_cfg_value(self.cfg, "weight", 1.0)),
                    "active_norm": str(_control_cfg_value(self.cfg, "active_norm", "sqrt")),
                    "chunk_delay": int(_control_cfg_value(self.cfg, "chunk_delay", 0)),
                    "log_every": int(_control_cfg_value(self.cfg, "log_every", 0)),
                    "freeze_root": bool(_control_cfg_value(self.cfg, "freeze_root", True)),
                }
                control_space = control_full.get("space", "absolute")
                cond_["y"]["guidance_fn"] = (
                    lambda x, freeze_root=True, betas=tar_beta[:, 0], round_l=round_l, control_space=control_space, **kwargs:
                    self._latent_delayed_window_joints(
                        x, betas, freeze_root, round_l, control_space, **kwargs
                    )
                )
                cond_["y"]["seed_update_fn"] = (
                    lambda x, seed, first_seed=first_seed:
                    self._update_dynamic_seeds(x, first_seed)
                )

            sample = self.model(cond_)["latents"].squeeze(2).permute(0, 2, 1)
            full_latents = guidance_utils.stitch_chunk_latents(sample, self.cfg.pre_frames)
            rec_all_upper = full_latents[..., :128]
            rec_all_hands = full_latents[..., 128:256]
            rec_all_lower = full_latents[..., 256:]
        else:
            in_audio_onset_tmp = None
            in_word_tmp = None
            last_sample = None
            for i in range(0, roundt):
                if audio_onset is not None:
                    in_audio_onset_tmp = audio_onset[
                        :,
                        i * (16000 // 30 * round_l) : (i + 1) * (16000 // 30 * round_l)
                        + 16000 // 30 * self.cfg.pre_frames * vqvae_squeeze_scale,
                    ]
                if in_word is not None:
                    in_word_tmp = in_word[
                        :,
                        i * round_l : (i + 1) * round_l
                        + self.cfg.pre_frames * vqvae_squeeze_scale,
                    ]

                in_id_tmp = loaded_data["tar_id"][
                    :, i * round_l : (i + 1) * round_l + self.cfg.pre_frames
                ]
                in_seed_tmp = in_seed[
                    :,
                    i
                    * round_l
                    // vqvae_squeeze_scale : (i + 1)
                    * round_l
                    // vqvae_squeeze_scale
                    + self.cfg.pre_frames,
                ]

                if i == 0:
                    in_seed_tmp = in_seed_tmp[:, : self.cfg.pre_frames, :]
                else:
                    in_seed_tmp = last_sample[:, -self.cfg.pre_frames :, :]

                cond_ = {"y": {}}
                cond_["y"]["audio_onset"] = in_audio_onset_tmp
                cond_["y"]["word"] = in_word_tmp
                cond_["y"]["id"] = in_id_tmp
                cond_["y"]["seed"] = in_seed_tmp
                cond_["y"]["style_feature"] = torch.zeros([bs, 512], device=self.mean_upper.device)

                sample = self.model(cond_)["latents"]
                sample = sample.squeeze(2).permute(0, 2, 1)
                last_sample = sample.clone()

                code_dim = self.vq_model_upper.code_dim
                rec_latent_upper = sample[..., :code_dim]
                rec_latent_hands = sample[..., code_dim : code_dim * 2]
                rec_latent_lower = sample[..., code_dim * 2 : code_dim * 3]

                if i == 0:
                    rec_all_upper.append(rec_latent_upper)
                    rec_all_hands.append(rec_latent_hands)
                    rec_all_lower.append(rec_latent_lower)
                else:
                    rec_all_upper.append(rec_latent_upper[:, self.cfg.pre_frames :])
                    rec_all_hands.append(rec_latent_hands[:, self.cfg.pre_frames :])
                    rec_all_lower.append(rec_latent_lower[:, self.cfg.pre_frames :])

            rec_all_upper = torch.cat(rec_all_upper, dim=1)
            rec_all_hands = torch.cat(rec_all_hands, dim=1)
            rec_all_lower = torch.cat(rec_all_lower, dim=1)

        if isinstance(rec_all_upper, list):
            rec_all_upper = torch.cat(rec_all_upper, dim=1)
            rec_all_hands = torch.cat(rec_all_hands, dim=1)
            rec_all_lower = torch.cat(rec_all_lower, dim=1)

        rec_all_upper = rec_all_upper * 5
        rec_all_hands = rec_all_hands * 5
        rec_all_lower = rec_all_lower * 5

        rec_upper = self.vq_model_upper.decode_continuous(rec_all_upper)
        rec_hands = self.vq_model_hands.decode_continuous(rec_all_hands)
        rec_lower = self.vq_model_lower.decode_continuous(rec_all_lower)

        rec_trans_v = rec_lower[..., -3:]
        rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
        rec_trans = torch.zeros_like(rec_trans_v)
        rec_trans = torch.cumsum(rec_trans_v, dim=-2)
        rec_trans[..., 1] = rec_trans_v[..., 1]
        rec_lower = rec_lower[..., :-3]

        rec_upper = rec_upper * self.std_upper + self.mean_upper
        rec_hands = rec_hands * self.std_hands + self.mean_hands
        rec_lower = rec_lower * self.std_lower + self.mean_lower

        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        if hasattr(self.cfg.model, "use_exp") and self.cfg.model.use_exp:
            rec_exps = tar_exps  # fallback to tar_exps since rec_face is not defined
        else:
            rec_exps = tar_exps

        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(
            rec_pose_upper, self.joint_mask_upper, bs * n
        )
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)

        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(
            rec_pose_lower, self.joint_mask_lower, bs * n
        )
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(
            rec_pose_hands, self.joint_mask_hands, bs * n
        )
        rec_pose = (
            rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        )
        rec_pose[:, 66:69] = tar_pose.reshape(bs * n, 55 * 3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j * 6)

        return {
            "rec_pose": rec_pose,
            "rec_exps": rec_exps,
            "rec_trans": rec_trans,
            "tar_pose": tar_pose,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_trans": tar_trans,
            "control_hint": control_full["hint"] if control_full is not None else None,
            "control_mask": control_full["mask"] if control_full is not None else None,
            "control_space": control_full.get("space", "absolute") if control_full is not None else "absolute",
            "control_setting": control_setting,
        }

    def train(self, epoch):

        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start

            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, "train", epoch)

            g_loss_final.backward()
            if self.cfg.solver.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.solver.grad_norm
                )
            self.opt.step()

            mem_cost = torch.cuda.memory_cached() / 1e9
            lr_g = self.opt.param_groups[0]["lr"]

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.cfg.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)
            if self.cfg.debug:
                if its == 1:
                    break
        self.opt_s.step(epoch)

    @torch.no_grad()
    def _common_test_inference(
        self,
        data_loader,
        epoch,
        mode="val",
        max_iterations=None,
        save_results=False,
        control_setting=None,
        control_rng=None,
        control_settings=None,
        control_seed=None,
        only_iteration=None,
    ):
        """
        Common inference logic shared by val, test, test_clip, and test_render methods.

        Args:
            data_loader: The data loader to iterate over
            epoch: Current epoch number
            mode: Mode string for logging ("val", "test", "test_clip", "test_render")
            max_iterations: Maximum number of iterations (None for no limit)
            save_results: Whether to save result files

        Returns:
            Dictionary containing computed metrics and results
        """
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0
        align_gt = 0
        latent_out = []
        latent_ori = []
        l2_all = 0
        lvel = 0
        skating_metrics_sum = {name: 0.0 for name in SKATING_METRIC_NAMES}
        gt_skating_metrics_sum = {name: 0.0 for name in SKATING_METRIC_NAMES}
        skating_count = 0
        control_metric_sums = {name: 0.0 for name in CONTROL_METRIC_NAMES}
        control_metric_counts = {name: 0 for name in CONTROL_METRIC_NAMES}
        num_sequences = 0
        l1_calculator_gt = metric.L1div()
        if hasattr(self, "l1_calculator"):
            self.l1_calculator.reset()
        results = []

        # Setup save path for test mode
        results_save_path = None
        if save_results:
            results_save_path = self.checkpoint_path + f"/{epoch}/"
            if mode == "test_render":
                if os.path.exists(results_save_path):
                    import shutil

                    shutil.rmtree(results_save_path)
            os.makedirs(results_save_path, exist_ok=True)

        self.model.eval()
        self.smplx.eval()
        if hasattr(self, "eval_copy"):
            self.eval_copy.eval()

        with torch.no_grad():
            iterator = enumerate(data_loader)
            if mode in ["test_clip", "test", "control_eval"]:
                iterator = enumerate(
                    tqdm(data_loader, desc=f"Testing {mode}", leave=True)
                )

            for its, batch_data in iterator:
                if only_iteration is not None:
                    if its < only_iteration:
                        continue
                    if its > only_iteration:
                        break
                elif max_iterations is not None and its >= max_iterations:
                    break

                sample_control_setting = control_setting
                sample_control_rng = control_rng
                if control_settings is not None:
                    if its >= len(control_settings):
                        break
                    sample_control_setting = control_settings[its]
                    sample_control_rng = np.random.default_rng(int(control_seed or 0) + its)
                    seq_id = None
                    if its < len(test_seq_list):
                        seq_id = test_seq_list.iloc[its].get("id", None)
                    seq_info = f" sample_id={seq_id}" if seq_id is not None else ""
                    logger.info(
                        "Control eval sample: "
                        f"sample_index={its}{seq_info} setting={sample_control_setting['name']} "
                        f"joints={list(sample_control_setting['joints'])} "
                        f"density={sample_control_setting['density']} "
                        f"space={sample_control_setting.get('space', 'absolute')}"
                    )

                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(
                    loaded_data,
                    control_setting=sample_control_setting,
                    control_rng=sample_control_rng,
                )

                tar_pose = net_out["tar_pose"]
                rec_pose = net_out["rec_pose"]
                tar_exps = net_out["tar_exps"]
                tar_beta = net_out["tar_beta"]
                rec_trans = net_out["rec_trans"]
                tar_trans = net_out.get("tar_trans", rec_trans)
                rec_exps = net_out.get("rec_exps", tar_exps)

                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # Handle frame rate conversion
                if (30 / self.cfg.data.pose_fps) != 1:
                    assert 30 % self.cfg.data.pose_fps == 0
                    n *= int(30 / self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(
                        tar_pose.permute(0, 2, 1),
                        scale_factor=30 / self.cfg.data.pose_fps,
                        mode="linear",
                    ).permute(0, 2, 1)
                    scale_factor = (
                        30 / self.cfg.data.pose_fps
                        if mode != "test"
                        else 30 / self.cfg.pose_fps
                    )
                    rec_pose = torch.nn.functional.interpolate(
                        rec_pose.permute(0, 2, 1),
                        scale_factor=scale_factor,
                        mode="linear",
                    ).permute(0, 2, 1)

                # Calculate latent representations for evaluation
                if hasattr(self, "eval_copy") and mode != "test_render":
                    remain = n % self.cfg.vae_test_len
                    latent_out.append(
                        self.eval_copy.map2latent(rec_pose[:, : n - remain])
                        .reshape(-1, self.cfg.vae_length)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    latent_ori.append(
                        self.eval_copy.map2latent(tar_pose[:, : n - remain])
                        .reshape(-1, self.cfg.vae_length)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs * n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs * n, j * 3)
                rec_root_rot_np = (
                    rc.axis_angle_to_matrix(rec_pose[:, :3])
                    .reshape(bs, n, 3, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Generate SMPLX vertices and joints in the translation-free frame used
                # by the existing BC/L1Div metrics.
                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=rec_trans.reshape(bs * n, 3) - rec_trans.reshape(bs * n, 3),
                    expression=tar_exps.reshape(bs * n, 100)
                    - tar_exps.reshape(bs * n, 100),
                    jaw_pose=rec_pose[:, 66:69],
                    global_orient=rec_pose[:, :3],
                    body_pose=rec_pose[:, 3 : 21 * 3 + 3],
                    left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3],
                    right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3],
                    return_joints=True,
                    leye_pose=rec_pose[:, 69:72],
                    reye_pose=rec_pose[:, 72:75],
                )
                vertices_tar = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=tar_trans.reshape(bs * n, 3) - tar_trans.reshape(bs * n, 3),
                    expression=tar_exps.reshape(bs * n, 100)
                    - tar_exps.reshape(bs * n, 100),
                    jaw_pose=tar_pose[:, 66:69],
                    global_orient=tar_pose[:, :3],
                    body_pose=tar_pose[:, 3 : 21 * 3 + 3],
                    left_hand_pose=tar_pose[:, 25 * 3 : 40 * 3],
                    right_hand_pose=tar_pose[:, 40 * 3 : 55 * 3],
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72],
                    reye_pose=tar_pose[:, 72:75],
                )

                joints_rec_full = (
                    vertices_rec["joints"]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(bs, n, 127, 3)[:, :n]
                )
                joints_tar_full = (
                    vertices_tar["joints"]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(bs, n, 127, 3)[:, :n]
                )
                joints_rec_all = joints_rec_full[:, :, :55]
                joints_tar_all = joints_tar_full[:, :, :55]
                joints_rec = joints_rec_all[0].reshape(n, 55 * 3)
                joints_tar = joints_tar_all[0].reshape(n, 55 * 3)

                # Calculate L1 diversity. L1div mutates its input, so pass copies.
                if hasattr(self, "l1_calculator"):
                    _ = self.l1_calculator.run(joints_rec.copy())
                    _ = l1_calculator_gt.run(joints_tar.copy())

                if mode != "test_render":
                    rec_trans_np = rec_trans.reshape(bs, n, 3).detach().cpu().numpy()
                    tar_trans_np = tar_trans.reshape(bs, n, 3).detach().cpu().numpy()
                    skating_metrics, _ = metric.calculate_foot_skating_metrics(
                        joints_rec_full + rec_trans_np[:, :, None, :],
                        fps=self.cfg.data.pose_fps,
                    )
                    gt_skating_metrics, _ = metric.calculate_foot_skating_metrics(
                        joints_tar_full + tar_trans_np[:, :, None, :],
                        fps=self.cfg.data.pose_fps,
                    )
                    for metric_name, source_name in SKATING_METRIC_MAP:
                        skating_metrics_sum[metric_name] += float(
                            np.sum(skating_metrics[source_name])
                        )
                        gt_skating_metrics_sum[metric_name] += float(
                            np.sum(gt_skating_metrics[source_name])
                        )
                    skating_count += skating_metrics["ratio_050"].shape[0]

                    control_hint = net_out.get("control_hint")
                    control_mask = net_out.get("control_mask")
                    if control_hint is not None and control_mask is not None:
                        gen_global_joints = joints_rec_all + rec_trans_np[:, :, None, :]
                        control_space = net_out.get("control_space", "absolute")
                        gen_global_joints = self._to_control_space(
                            gen_global_joints,
                            control_space,
                            rec_root_rot_np if control_space == "relative" else None,
                        )
                        control_metrics = metric.calculate_control_error_metrics(
                            gen_global_joints,
                            control_hint.detach().cpu().numpy(),
                            control_mask.detach().cpu().numpy(),
                            threshold_m=self._control_eval_threshold_m(),
                            chunk_length=self.cfg.data.pose_length,
                            chunk_step=self.cfg.data.pose_length
                            - self.cfg.pre_frames * self.cfg.vqvae_squeeze_scale,
                            chunk_seed_frames=self.cfg.pre_frames * self.cfg.vqvae_squeeze_scale,
                        )
                        active_chunks = int(control_metrics["active_chunks"])
                        active_points = int(control_metrics["active_points"])
                        if active_chunks > 0:
                            for metric_name in CONTROL_TRAJ_METRIC_NAMES:
                                control_metric_sums[metric_name] += (
                                    control_metrics[metric_name] * active_chunks
                                )
                                control_metric_counts[metric_name] += active_chunks
                        if active_points > 0:
                            for metric_name in (*CONTROL_LOC_METRIC_NAMES, "avg_err_cm"):
                                control_metric_sums[metric_name] += (
                                    control_metrics[metric_name] * active_points
                                )
                                control_metric_counts[metric_name] += active_points

                # Calculate alignment for single batch
                if (
                    hasattr(self, "alignmenter")
                    and self.alignmenter is not None
                    and bs == 1
                    and mode != "test_render"
                ):
                    in_audio_eval, sr = librosa.load(
                        self.cfg.data.data_path
                        + "wave16k/"
                        + test_seq_list.iloc[its]["id"]
                        + ".wav"
                    )
                    in_audio_eval = librosa.resample(
                        in_audio_eval, orig_sr=sr, target_sr=self.cfg.data.audio_sr
                    )
                    a_offset = int(
                        self.align_mask
                        * (self.cfg.data.audio_sr / self.cfg.data.pose_fps)
                    )
                    onset_bt = self.alignmenter.load_audio(
                        in_audio_eval[
                            : int(self.cfg.data.audio_sr / self.cfg.data.pose_fps * n)
                        ],
                        a_offset,
                        len(in_audio_eval) - a_offset,
                        True,
                    )
                    beat_vel = self.alignmenter.load_pose(
                        joints_rec, self.align_mask, n - self.align_mask, 30, True
                    )
                    align += self.alignmenter.calculate_align(
                        onset_bt, beat_vel, 30
                    ) * (n - 2 * self.align_mask)
                    beat_vel_gt = self.alignmenter.load_pose(
                        joints_tar, self.align_mask, n - self.align_mask, 30, True
                    )
                    align_gt += self.alignmenter.calculate_align(
                        onset_bt, beat_vel_gt, 30
                    ) * (n - 2 * self.align_mask)

                # Mode-specific processing
                if mode == "test" and save_results:
                    # Calculate facial losses for test mode
                    vertices_rec_face = self.smplx(
                        betas=tar_beta.reshape(bs * n, 300),
                        transl=rec_trans.reshape(bs * n, 3)
                        - rec_trans.reshape(bs * n, 3),
                        expression=rec_exps.reshape(bs * n, 100),
                        jaw_pose=rec_pose[:, 66:69],
                        global_orient=rec_pose[:, :3] - rec_pose[:, :3],
                        body_pose=rec_pose[:, 3 : 21 * 3 + 3]
                        - rec_pose[:, 3 : 21 * 3 + 3],
                        left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3]
                        - rec_pose[:, 25 * 3 : 40 * 3],
                        right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3]
                        - rec_pose[:, 40 * 3 : 55 * 3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72] - rec_pose[:, 69:72],
                        reye_pose=rec_pose[:, 72:75] - rec_pose[:, 72:75],
                    )
                    vertices_tar_face = self.smplx(
                        betas=tar_beta.reshape(bs * n, 300),
                        transl=tar_trans.reshape(bs * n, 3)
                        - tar_trans.reshape(bs * n, 3),
                        expression=tar_exps.reshape(bs * n, 100),
                        jaw_pose=tar_pose[:, 66:69],
                        global_orient=tar_pose[:, :3] - tar_pose[:, :3],
                        body_pose=tar_pose[:, 3 : 21 * 3 + 3]
                        - tar_pose[:, 3 : 21 * 3 + 3],
                        left_hand_pose=tar_pose[:, 25 * 3 : 40 * 3]
                        - tar_pose[:, 25 * 3 : 40 * 3],
                        right_hand_pose=tar_pose[:, 40 * 3 : 55 * 3]
                        - tar_pose[:, 40 * 3 : 55 * 3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose[:, 69:72] - tar_pose[:, 69:72],
                        reye_pose=tar_pose[:, 72:75] - tar_pose[:, 72:75],
                    )

                    facial_rec = (
                        vertices_rec_face["vertices"].reshape(1, n, -1)[0, :n].cpu()
                    )
                    facial_tar = (
                        vertices_tar_face["vertices"].reshape(1, n, -1)[0, :n].cpu()
                    )
                    face_vel_loss = self.vel_loss(
                        facial_rec[1:, :] - facial_tar[:-1, :],
                        facial_tar[1:, :] - facial_tar[:-1, :],
                    )
                    l2 = self.reclatent_loss(facial_rec, facial_tar)
                    l2_all += l2.item() * n
                    lvel += face_vel_loss.item() * n

                # Save results if needed
                if save_results:
                    if mode == "test":
                        # Save NPZ files for test mode
                        tar_pose_np = tar_pose.detach().cpu().numpy()
                        rec_pose_np = rec_pose.detach().cpu().numpy()
                        rec_trans_np = (
                            rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )
                        rec_exp_np = (
                            rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )
                        tar_exp_np = (
                            tar_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )
                        tar_trans_np = (
                            tar_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )

                        gt_npz = np.load(
                            self.cfg.data.data_path
                            + self.cfg.data.pose_rep
                            + "/"
                            + test_seq_list.iloc[its]["id"]
                            + ".npz",
                            allow_pickle=True,
                        )

                        np.savez(
                            results_save_path
                            + "gt_"
                            + test_seq_list.iloc[its]["id"]
                            + ".npz",
                            betas=gt_npz["betas"],
                            poses=tar_pose_np,
                            expressions=tar_exp_np,
                            trans=tar_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )
                        np.savez(
                            results_save_path
                            + "res_"
                            + test_seq_list.iloc[its]["id"]
                            + ".npz",
                            betas=gt_npz["betas"],
                            poses=rec_pose_np,
                            expressions=rec_exp_np,
                            trans=rec_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )

                    elif mode == "test_render":
                        # Save results and render for test_render mode
                        audio_name = loaded_data["audio_name"][0]
                        rec_pose_np = rec_pose.detach().cpu().numpy()
                        rec_trans_np = (
                            rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )
                        rec_exp_np = (
                            rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )

                        gt_npz = np.load(
                            "./demo/examples/2_scott_0_1_1.npz", allow_pickle=True
                        )
                        file_name = audio_name.split("/")[-1].split(".")[0]
                        results_npz_file_save_path = (
                            results_save_path + f"result_{file_name}.npz"
                        )

                        np.savez(
                            results_npz_file_save_path,
                            betas=gt_npz["betas"],
                            poses=rec_pose_np,
                            expressions=rec_exp_np,
                            trans=rec_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )

                        render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                            results_npz_file_save_path,
                            results_save_path,
                            audio_name,
                            self.cfg.data_path_1 + "smplx_models/",
                            use_matplotlib=False,
                            args=self.cfg,
                        )

                total_length += n
                num_sequences += bs

        skating_metrics_avg = {
            name: skating_metrics_sum[name] / skating_count if skating_count > 0 else 0.0
            for name in SKATING_METRIC_NAMES
        }
        gt_skating_metrics_avg = {
            f"gt_{name}": gt_skating_metrics_sum[name] / skating_count
            if skating_count > 0 else 0.0
            for name in SKATING_METRIC_NAMES
        }

        control_metrics_avg = {
            name: control_metric_sums[name] / control_metric_counts[name]
            if control_metric_counts[name] > 0 else 0.0
            for name in CONTROL_METRIC_NAMES
        }
        control_metrics_avg["active_chunks"] = control_metric_counts[CONTROL_TRAJ_METRIC_NAMES[0]]

        return {
            "total_length": total_length,
            "num_sequences": num_sequences,
            "align": align,
            "align_gt": align_gt,
            "latent_out": latent_out,
            "latent_ori": latent_ori,
            "l2_all": l2_all,
            "lvel": lvel,
            "gt_l1div": l1_calculator_gt.avg() if l1_calculator_gt.counter > 0 else 0.0,
            "skating_metrics": skating_metrics_avg,
            "gt_skating_metrics": gt_skating_metrics_avg,
            "control_metrics": control_metrics_avg,
            "start_time": start_time,
        }

    def val(self, epoch):
        self.tracker.reset()

        results = self._common_test_inference(
            self.test_loader, epoch, mode="val", max_iterations=15
        )

        total_length = results["total_length"]
        num_sequences = results["num_sequences"]
        align = results["align"]
        align_gt = results["align_gt"]
        latent_out = results["latent_out"]
        latent_ori = results["latent_ori"]
        l2_all = results["l2_all"]
        lvel = results["lvel"]
        gt_l1div = results["gt_l1div"]
        skating_metrics = results["skating_metrics"]
        gt_skating_metrics = results["gt_skating_metrics"]
        start_time = results["start_time"]

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)

        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)
        gt_fgd = data_tools.FIDCalculator.frechet_distance(latent_ori_all, latent_ori_all)
        logger.info(f"gt fgd score: {gt_fgd}")
        self.tracker.update_meter("gt_fgd", "val", gt_fgd)

        align_denom = total_length - 2 * num_sequences * self.align_mask
        align_avg = align / align_denom
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)
        align_gt_avg = align_gt / align_denom
        logger.info(f"gt align score: {align_gt_avg}")
        self.tracker.update_meter("gt_bc", "val", align_gt_avg)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)
        logger.info(f"gt l1div score: {gt_l1div}")
        self.tracker.update_meter("gt_l1div", "val", gt_l1div)
        for metric_name in SKATING_METRIC_NAMES:
            logger.info(f"{metric_name} score: {skating_metrics[metric_name]}")
            self.tracker.update_meter(metric_name, "val", skating_metrics[metric_name])
            gt_metric_name = f"gt_{metric_name}"
            logger.info(f"{gt_metric_name} score: {gt_skating_metrics[gt_metric_name]}")
            self.tracker.update_meter(
                gt_metric_name, "val", gt_skating_metrics[gt_metric_name]
            )

        self.val_recording(epoch)

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test_clip(self, epoch):
        self.tracker.reset()

        # Test on CLIP dataset
        results_clip = self._common_test_inference(
            self.test_clip_loader, epoch, mode="test_clip"
        )

        total_length_clip = results_clip["total_length"]
        latent_out_clip = results_clip["latent_out"]
        latent_ori_clip = results_clip["latent_ori"]
        start_time = results_clip["start_time"]

        latent_out_all_clip = np.concatenate(latent_out_clip, axis=0)
        latent_ori_all_clip = np.concatenate(latent_ori_clip, axis=0)

        fgd_clip = data_tools.FIDCalculator.frechet_distance(
            latent_out_all_clip, latent_ori_all_clip
        )
        logger.info(f"test_clip fgd score: {fgd_clip}")
        self.tracker.update_meter("test_clip_fgd", "val", fgd_clip)

        current_time = time.time()
        test_clip_time = current_time - start_time
        logger.info(
            f"total test_clip inference time: {int(test_clip_time)} s for {int(total_length_clip/self.cfg.data.pose_fps)} s motion"
        )

        # Test on regular test dataset for recording
        results_test = self._common_test_inference(
            self.test_loader, epoch, mode="test_clip"
        )

        total_length = results_test["total_length"]
        num_sequences = results_test["num_sequences"]
        align = results_test["align"]
        align_gt = results_test["align_gt"]
        latent_out = results_test["latent_out"]
        latent_ori = results_test["latent_ori"]
        gt_l1div = results_test["gt_l1div"]
        skating_metrics = results_test["skating_metrics"]
        gt_skating_metrics = results_test["gt_skating_metrics"]

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)

        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)
        gt_fgd = data_tools.FIDCalculator.frechet_distance(latent_ori_all, latent_ori_all)
        logger.info(f"gt fgd score: {gt_fgd}")
        self.tracker.update_meter("gt_fgd", "val", gt_fgd)

        align_denom = total_length - 2 * num_sequences * self.align_mask
        align_avg = align / align_denom
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)
        align_gt_avg = align_gt / align_denom
        logger.info(f"gt align score: {align_gt_avg}")
        self.tracker.update_meter("gt_bc", "val", align_gt_avg)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)
        logger.info(f"gt l1div score: {gt_l1div}")
        self.tracker.update_meter("gt_l1div", "val", gt_l1div)
        for metric_name in SKATING_METRIC_NAMES:
            logger.info(f"{metric_name} score: {skating_metrics[metric_name]}")
            self.tracker.update_meter(metric_name, "val", skating_metrics[metric_name])
            gt_metric_name = f"gt_{metric_name}"
            logger.info(f"{gt_metric_name} score: {gt_skating_metrics[gt_metric_name]}")
            self.tracker.update_meter(
                gt_metric_name, "val", gt_skating_metrics[gt_metric_name]
            )

        self.val_recording(epoch)

        end_time = time.time() - current_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def _save_control_eval_artifacts(self, control_eval_dir):
        config_path = os.path.join(control_eval_dir, "resolved_config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=self.cfg, f=f)

        snapshot_dir = os.path.join(control_eval_dir, "code")
        os.makedirs(snapshot_dir, exist_ok=True)
        repo_root = os.getcwd()
        source_files = [
            "models/Diffusion.py",
            "trainer/generative_trainer.py",
            "utils/guidance.py",
            "utils/metric.py",
            "gen.py",
            "configs_new/diffusion_rvqvae_128.yaml",
        ]
        for rel_path in source_files:
            src = os.path.join(repo_root, rel_path)
            if not os.path.exists(src):
                continue
            dst = os.path.join(snapshot_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    def _run_control_evaluation(self, epoch):
        if not self._control_eval_enabled():
            return
        if self.cfg.model.g_name != "GestureDiffusion":
            raise ValueError("control_eval.enabled=True requires GestureDiffusion")

        seed = int(_control_cfg_value(self.cfg, "seed", 42))
        max_samples = int(_control_cfg_value(self.cfg, "max_samples", 15))
        eval_settings = list(self.control_eval_settings)
        if max_samples > 0:
            eval_settings = eval_settings[:max_samples]
        available_samples = len(self.test_data.selected_file)
        if available_samples > 0:
            eval_settings = eval_settings[:available_samples]
        if getattr(self.cfg, "debug", False):
            eval_settings = eval_settings[:1]

        control_log_sink = None
        if self.rank == 0:
            timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            run_name = str(_control_cfg_value(self.cfg, "name", "")).strip()
            run_name = run_name.replace(os.sep, "_")
            folder_name = timestamp if not run_name else f"{timestamp}__{run_name}"
            control_eval_dir = os.path.join(self.checkpoint_path, folder_name)
            os.makedirs(control_eval_dir, exist_ok=True)
            control_log_path = os.path.join(control_eval_dir, "control_eval.log")
            control_log_sink = logger.add(
                control_log_path,
                format="<blue>{time: MM-DD HH:mm:ss}</blue> | <level>{message}</level>",
            )
            self._save_control_eval_artifacts(control_eval_dir)
            logger.info(f"Control eval dir: {control_eval_dir}")
            logger.info(f"Control eval log: {control_log_path}")

        group_summaries = []
        seen_groups = set()
        for setting in eval_settings:
            group_name = setting["joint_group"]
            if group_name in seen_groups:
                continue
            seen_groups.add(group_name)
            group_summaries.append(f"{group_name}={list(setting['joints'])}")

        logger.info("Control eval configuration:")
        logger.info(f"joint combinations: {group_summaries}")
        logger.info(
            f"densities: {list(_control_cfg_value(self.cfg, 'densities', CONTROL_DENSITIES))} per chunk"
        )
        logger.info(f"control settings: {len(self.control_eval_settings)}")
        logger.info(f"available samples: {available_samples}")
        logger.info(f"max_samples total: {len(eval_settings)}")
        logger.info(
            f"space: {str(_control_cfg_value(self.cfg, 'space', 'absolute')).strip().lower()}"
        )
        logger.info(
            f"chunk_delay: {int(_control_cfg_value(self.cfg, 'chunk_delay', 0))}"
        )
        logger.info(
            f"iters_early: {int(_control_cfg_value(self.cfg, 'iters_early', 5))}"
        )
        logger.info(
            f"iters_late: {int(_control_cfg_value(self.cfg, 'iters_late', 30))}"
        )
        logger.info(
            f"late_start: {int(_control_cfg_value(self.cfg, 'late_start', 300))}"
        )
        logger.info(
            f"post_iters: {int(_control_cfg_value(self.cfg, 'post_iters', 0))}"
        )
        logger.info(
            f"scale: {float(_control_cfg_value(self.cfg, 'scale', 20.0))}"
        )
        logger.info(
            f"weight: {float(_control_cfg_value(self.cfg, 'weight', 1.0))}"
        )
        logger.info(
            f"active_norm: {str(_control_cfg_value(self.cfg, 'active_norm', 'sqrt'))}"
        )
        logger.info(
            f"freeze_root: {bool(_control_cfg_value(self.cfg, 'freeze_root', True))}"
        )
        logger.info(
            f"name: {str(_control_cfg_value(self.cfg, 'name', '')).strip()}"
        )

        try:
            generation_start_time = time.time()
            results = self._common_test_inference(
                self.test_loader,
                epoch,
                mode="control_eval",
                max_iterations=len(eval_settings),
                control_settings=eval_settings,
                control_seed=seed,
            )
            generation_time = time.time() - generation_start_time
            control_metrics = results["control_metrics"]
            total_length = results["total_length"]
            num_sequences = results["num_sequences"]
            logger.info(
                f"total generation time: {int(generation_time)} s "
                f"for {int(total_length / self.cfg.data.pose_fps)} s motion"
            )
            latent_out_all = np.concatenate(results["latent_out"], axis=0)
            latent_ori_all = np.concatenate(results["latent_ori"], axis=0)
            fgd = data_tools.FIDCalculator.frechet_distance(
                latent_out_all, latent_ori_all
            )
            gt_fgd = data_tools.FIDCalculator.frechet_distance(
                latent_ori_all, latent_ori_all
            )
            align_denom = total_length - 2 * num_sequences * self.align_mask
            align_avg = results["align"] / align_denom
            gt_align_avg = results["align_gt"] / align_denom
            metric_values = {
                "fgd": fgd,
                "gt_fgd": gt_fgd,
                "align": align_avg,
                "gt_align": gt_align_avg,
                "l1div": self.l1_calculator.avg(),
                "gt_l1div": results["gt_l1div"],
                "foot_skating_ratio": results["skating_metrics"]["skating_ratio"],
                **{name: control_metrics[name] for name in CONTROL_TRAJ_METRIC_NAMES},
                **{name: control_metrics[name] for name in CONTROL_LOC_METRIC_NAMES},
                "avg_err_cm": control_metrics["avg_err_cm"],
            }
            for metric_name, value in metric_values.items():
                tracker_name = f"control/aggregate/{metric_name}"
                logger.info(f"{tracker_name}: {value}")
                self.test_recording(tracker_name, value, epoch)
        finally:
            if control_log_sink is not None:
                logger.remove(control_log_sink)

    def test(self, epoch):
        if self._control_eval_enabled():
            start_time = time.time()
            self._run_control_evaluation(epoch)
            end_time = time.time() - start_time
            logger.info(f"total control eval time: {int(end_time)} s")
            return

        results_save_path = self.checkpoint_path + f"/{epoch}/"
        os.makedirs(results_save_path, exist_ok=True)

        results = self._common_test_inference(
            self.test_loader, epoch, mode="test", save_results=True
        )

        total_length = results["total_length"]
        num_sequences = results["num_sequences"]
        align = results["align"]
        align_gt = results["align_gt"]
        latent_out = results["latent_out"]
        latent_ori = results["latent_ori"]
        l2_all = results["l2_all"]
        lvel = results["lvel"]
        gt_l1div = results["gt_l1div"]
        skating_metrics = results["skating_metrics"]
        gt_skating_metrics = results["gt_skating_metrics"]
        start_time = results["start_time"]

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.test_recording("fgd", fgd, epoch)
        gt_fgd = data_tools.FIDCalculator.frechet_distance(latent_ori_all, latent_ori_all)
        logger.info(f"gt fgd score: {gt_fgd}")
        self.test_recording("gt_fgd", gt_fgd, epoch)

        align_denom = total_length - 2 * num_sequences * self.align_mask
        align_avg = align / align_denom
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)
        align_gt_avg = align_gt / align_denom
        logger.info(f"gt align score: {align_gt_avg}")
        self.test_recording("gt_bc", align_gt_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)
        logger.info(f"gt l1div score: {gt_l1div}")
        self.test_recording("gt_l1div", gt_l1div, epoch)
        for metric_name in SKATING_METRIC_NAMES:
            logger.info(f"{metric_name} score: {skating_metrics[metric_name]}")
            self.test_recording(metric_name, skating_metrics[metric_name], epoch)
            gt_metric_name = f"gt_{metric_name}"
            logger.info(f"{gt_metric_name} score: {gt_skating_metrics[gt_metric_name]}")
            self.test_recording(
                gt_metric_name, gt_skating_metrics[gt_metric_name], epoch
            )

        self._run_control_evaluation(epoch)

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test_render(self, epoch):
        import platform

        if platform.system() == "Linux":
            os.environ["PYOPENGL_PLATFORM"] = "egl"

        """
        input audio and text, output motion
        do not calculate loss and metric
        save video
        """
        results = self._common_test_inference(
            self.test_loader, epoch, mode="test_render", save_results=True
        )

    def load_checkpoint(self, checkpoint):
        # checkpoint is already a dict, do NOT call torch.load again!
        try:
            ckpt_state_dict = checkpoint["model_state_dict"]
        except:
            ckpt_state_dict = checkpoint["model_state"]
        # remove 'audioEncoder' from the state_dict due to legacy issues
        ckpt_state_dict = {
            k: v
            for k, v in ckpt_state_dict.items()
            if "modality_encoder.audio_encoder." not in k
        }
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        try:
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        except:
            print("No optimizer loaded!")
        if (
            "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.opt_s.load_state_dict(checkpoint["scheduler_state_dict"])
        if "val_best" in checkpoint:
            self.val_best = checkpoint["val_best"]
        logger.info("Checkpoint loaded successfully.")
