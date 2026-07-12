import argparse
import json
import logging
import os
import sys
import time
import warnings
import re
import faulthandler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloaders import data_tools
from dataloaders.mix_sep import CustomDataset
from models.vq.encdec import Decoder
from models.vq.model import RVQVAE
from utils import other_tools
from utils.config import parse_args
from utils.joints import hands_body_mask, lower_body_mask, upper_body_mask

warnings.filterwarnings("ignore")
faulthandler.enable()


def get_logger(out_dir):
    logger = logging.getLogger("UnifiedRVQVAE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_hdlr = logging.FileHandler(os.path.join(out_dir, "run.log"))
    file_hdlr.setFormatter(formatter)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super().__init__()
        if recons_loss == "l1":
            self.loss = nn.L1Loss()
        elif recons_loss == "l2":
            self.loss = nn.MSELoss()
        elif recons_loss == "l1_smooth":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown reconstruction loss: {recons_loss}")

    def forward(self, motion_pred, motion_gt):
        return self.loss(motion_pred, motion_gt)


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Train a unified decoder from frozen upper/hands/lower RVQ-VAE latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/diffuser_rvqvae_128.yaml")
    parser.add_argument("--dataset-config", default="configs/beat2_rvqvae.yaml")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--total-iter", default=80000, type=int)
    parser.add_argument("--warm-up-iter", default=400, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lr-scheduler", default=[50000, 200000, 400000], nargs="+", type=int)
    parser.add_argument("--gamma", default=0.05, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--recons-loss", default="l1_smooth", choices=["l1", "l2", "l1_smooth"])
    parser.add_argument("--loss-vel", default=0.5, type=float)
    parser.add_argument("--out-dir", default="outputs/rvqvae_simple")
    parser.add_argument("--exp-name", default="UnifiedRVQVAE")
    parser.add_argument("--print-iter", default=200, type=int)
    parser.add_argument("--eval-iter", default=100, type=int)
    parser.add_argument("--loader-workers", default=0, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--resume-pth", default=None)
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    return parser.parse_args()


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


def build_full_body_trans_mask():
    whole_body_mask = []
    for joint in list(range(0, 22)) + list(range(25, 55)):
        whole_body_mask.extend([joint * 6 + i for i in range(6)])
    whole_body_mask.extend([330, 331, 332])
    return whole_body_mask


def build_rvqvae(args, dim_pose, ckpt_path, device):
    model = RVQVAE(
        args,
        dim_pose,
        args.nb_code,
        args.code_dim,
        args.code_dim,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["net"], strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class FrozenPartTokenizer(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.upper = build_rvqvae(args, 78, args.vqvae_upper_path, device)
        self.hands = build_rvqvae(args, 180, args.vqvae_hands_path, device)
        self.lower = build_rvqvae(args, 57, args.vqvae_lower_path, device)
        self.latent_scale = args.vqvae_latent_scale

    @staticmethod
    def quantized_latent(model, x):
        x_in = model.preprocess(x)
        encoded = model.encoder(x_in)
        quantized, _, _, _ = model.quantizer(encoded, sample_codebook_temp=0.0)
        return quantized

    @torch.no_grad()
    def forward(self, motion):
        upper = motion[..., upper_body_mask]
        hands = motion[..., hands_body_mask]
        lower = motion[..., lower_body_mask]
        trans_v = motion[..., 330:333]
        lower = torch.cat([lower, trans_v], dim=-1)

        z_upper = self.quantized_latent(self.upper, upper)
        z_hands = self.quantized_latent(self.hands, hands)
        z_lower = self.quantized_latent(self.lower, lower)
        latents = torch.cat([z_upper, z_hands, z_lower], dim=1)
        return latents / self.latent_scale


class UnifiedDecoder(nn.Module):
    def __init__(self, args, input_width=315, latent_width=384):
        super().__init__()
        self.latent_scale = args.vqvae_latent_scale
        self.decoder = Decoder(
            input_width,
            latent_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            activation=args.vq_act,
            norm=args.vq_norm,
        )

    def forward(self, latents):
        return self.decoder(latents * self.latent_scale)


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def load_eval_model(dataset_args, device):
    eval_model_module = __import__("models.motion_representation", fromlist=["something"])
    dataset_args.vae_layer = 4
    dataset_args.vae_length = 240
    dataset_args.vae_test_dim = 330
    dataset_args.variational = False
    dataset_args.data_path_1 = "./datasets/hub/"
    dataset_args.vae_grow = [1, 1, 2, 1]
    eval_copy = getattr(eval_model_module, "VAESKConv")(dataset_args).to(device)

    eval_path = os.path.join(dataset_args.data_path, dataset_args.e_path)
    if not os.path.exists(eval_path):
        eval_path = dataset_args.e_path
    other_tools.load_checkpoints(eval_copy, eval_path, "VAESKConv")
    eval_copy.eval()
    return eval_copy


def evaluate(decoder, tokenizer, test_loader, eval_copy, mean_pose, std_pose, full_mask, out_dir,
             nb_iter, fid_his, l2_his, device, logger):
    decoder.eval()
    latent_out = []
    latent_ori = []
    diffs = []
    l2_all = 0.0

    with torch.no_grad():
        for batch_data in test_loader:
            gt_full = batch_data.to(device).float()
            n = gt_full.shape[1]
            remain = n % 8
            if remain:
                gt_full = gt_full[:, :-remain]
                n = gt_full.shape[1]

            gt_motion = gt_full[..., full_mask]
            latents = tokenizer(gt_full)
            pred_motion = decoder(latents)
            diff = pred_motion - gt_motion

            rec_motion = gt_full.clone()
            rec_motion[..., full_mask] = pred_motion
            rec_pose = rec_motion[..., :330] * std_pose + mean_pose
            gt_pose = gt_full[..., :330] * std_pose + mean_pose

            remain = n % 32
            if remain:
                rec_pose = rec_pose[:, :-remain]
                gt_pose = gt_pose[:, :-remain]
                diff_for_log = diff[:, :-remain]
            else:
                diff_for_log = diff

            rec_latent = eval_copy.map2latent(rec_pose)
            gt_latent = eval_copy.map2latent(gt_pose)
            latent_out.append(rec_latent.reshape(-1, rec_latent.shape[-1]).detach().cpu().numpy())
            latent_ori.append(gt_latent.reshape(-1, gt_latent.shape[-1]).detach().cpu().numpy())
            diffs.append(diff_for_log.reshape(-1, diff_for_log.shape[-1]).detach().cpu().numpy())
            l2_all += torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()

    latent_out_all = np.concatenate(latent_out, axis=0)
    latent_ori_all = np.concatenate(latent_ori, axis=0)
    np.concatenate(diffs, axis=0)

    fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
    l2 = l2_all / len(test_loader)
    fid_his.append(fid)
    l2_his.append(l2)
    logger.info(f"Eval. Iter {nb_iter}: fid {fid:.6f} \t L2 {l2:.6f}")

    import matplotlib.pyplot as plt
    plt.figure()
    iterations = [i * nb_iter for i in range(len(fid_his))]
    plt.plot(iterations, fid_his, label="FID")
    plt.plot(iterations, l2_his, label="L2")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("FID and L2 over Iterations")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "fid_l2_plot.png"))
    plt.close()
    decoder.train()
    return fid, l2


def infer_iter_from_checkpoint_path(path):
    match = re.search(r"unified_decoder_(\d+)\.pth$", os.path.basename(path or ""))
    return int(match.group(1)) if match else 0

def save_checkpoint(decoder, optimizer, scheduler, dataset_args, script_args, full_mask, out_dir, nb_iter):
    ckpt = {
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iter": nb_iter,
        "metadata": {
            "vqvae_upper_path": dataset_args.vqvae_upper_path,
            "vqvae_hands_path": dataset_args.vqvae_hands_path,
            "vqvae_lower_path": dataset_args.vqvae_lower_path,
            "latent_dim": 384,
            "output_dim": 315,
            "output_mask": full_mask,
            "vqvae_latent_scale": dataset_args.vqvae_latent_scale,
            "train_args": vars(script_args),
        },
    }
    torch.save(ckpt, os.path.join(out_dir, f"unified_decoder_{nb_iter}.pth"))
    torch.save(ckpt, os.path.join(out_dir, "latest.pth"))


def assert_frozen(module):
    trainable = [name for name, param in module.named_parameters() if param.requires_grad]
    if trainable:
        raise RuntimeError(f"Frozen tokenizer has trainable params: {trainable[:5]}")


def main():
    script_args = get_args_parser()
    torch.manual_seed(script_args.seed)

    out_dir = os.path.join(script_args.out_dir, script_args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir)
    logger.info(json.dumps(vars(script_args), indent=4, sort_keys=True))

    dataset_args, _ = parse_args(script_args.config)
    data_args, _ = parse_args(script_args.dataset_config)
    for key in (
        "vqvae_upper_path",
        "vqvae_hands_path",
        "vqvae_lower_path",
        "use_trans",
        "vqvae_latent_scale",
        "mean_pose_path",
        "std_pose_path",
        "mean_trans_path",
        "std_trans_path",
        "nb_code",
        "code_dim",
        "down_t",
        "stride_t",
        "width",
        "depth",
        "dilation_growth_rate",
        "vq_act",
        "vq_norm",
        "num_quantizers",
        "shared_codebook",
        "quantize_dropout_prob",
        "mu",
    ):
        setattr(data_args, key, getattr(dataset_args, key))

    logger.info("Building train dataset")
    train_set = CustomDataset(data_args, "train", build_cache=True)
    logger.info(f"Train dataset samples: {len(train_set)}")
    logger.info("Building test dataset")
    test_set = CustomDataset(data_args, "test", build_cache=True)
    logger.info(f"Test dataset samples: {len(test_set)}")
    logger.info(f"Creating dataloaders with num_workers={script_args.loader_workers}")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        script_args.batch_size,
        shuffle=True,
        num_workers=script_args.loader_workers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        1,
        shuffle=False,
        num_workers=script_args.loader_workers,
        drop_last=True,
    )
    train_loader_iter = cycle(train_loader)
    logger.info("Prefetching smoke batch before CUDA model construction")
    smoke_cpu = next(train_loader_iter).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Building frozen RVQ-VAE tokenizer")
    tokenizer = FrozenPartTokenizer(data_args, device)
    assert_frozen(tokenizer)
    logger.info("Frozen tokenizer loaded")
    logger.info("Building unified decoder")
    decoder = UnifiedDecoder(data_args).to(device)
    full_mask = build_full_body_trans_mask()

    with torch.no_grad():
        logger.info("Running smoke shape check")
        smoke = smoke_cpu.to(device)
        z = tokenizer(smoke)
        pred = decoder(z)
        target = smoke[..., full_mask]
        assert z.shape[1] == 384, z.shape
        assert pred.shape == target.shape, (pred.shape, target.shape)
        logger.info(f"Smoke shapes: latents={tuple(z.shape)} pred={tuple(pred.shape)}")

    writer = SummaryWriter(out_dir)
    logger.info("Loading VAESKConv eval model")
    eval_copy = load_eval_model(data_args, device)
    for stat_path in (data_args.mean_pose_path, data_args.std_pose_path):
        if not os.path.exists(stat_path):
            raise FileNotFoundError(f"Required normalization file not found: {stat_path}")
    mean_pose = torch.from_numpy(np.load(data_args.mean_pose_path)).to(device)
    std_pose = torch.from_numpy(np.load(data_args.std_pose_path)).to(device)

    loss_fn = ReConsLoss(script_args.recons_loss)
    optimizer = optim.AdamW(
        decoder.parameters(),
        lr=script_args.lr,
        betas=(0.9, 0.99),
        weight_decay=script_args.weight_decay,
    )
    start_iter = 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=script_args.lr_scheduler,
        gamma=script_args.gamma,
    )

    if script_args.resume_pth:
        logger.info(f"loading checkpoint from {script_args.resume_pth}")
        ckpt = torch.load(script_args.resume_pth, map_location="cpu")
        decoder.load_state_dict(ckpt["decoder"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = int(ckpt.get("iter", infer_iter_from_checkpoint_path(script_args.resume_pth)))
        logger.info(f"Resumed unified decoder training from iter {start_iter}")

    if script_args.mode == "eval":
        fid_his, l2_his = [], []
        evaluate(decoder, tokenizer, test_loader, eval_copy, mean_pose, std_pose, full_mask,
                 out_dir, 0, fid_his, l2_his, device, logger)
        return

    decoder.train()
    warmup_start_time = time.time()
    avg_recons = 0.0
    for nb_iter in range(1, script_args.warm_up_iter) if start_iter == 0 else ():
        optimizer, current_lr = update_lr_warm_up(
            optimizer,
            nb_iter,
            script_args.warm_up_iter,
            script_args.lr,
        )
        gt_full = next(train_loader_iter).to(device).float()
        gt_motion = gt_full[..., full_mask]
        latents = tokenizer(gt_full)
        pred_motion = decoder(latents)
        loss_motion = loss_fn(pred_motion, gt_motion)
        loss = loss_motion

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_recons += loss_motion.item()

        if nb_iter % script_args.print_iter == 0:
            avg_recons /= script_args.print_iter
            elapsed = time.time() - warmup_start_time
            iter_time = elapsed / nb_iter
            eta = iter_time * max(0, script_args.warm_up_iter - nb_iter)
            logger.info(
                f"Warmup. Iter {nb_iter}/{script_args.warm_up_iter}: lr {current_lr:.5f} \t Recons. {avg_recons:.5f} \t {iter_time:.3f}s/it \t elapsed {format_duration(elapsed)} \t eta {format_duration(eta)}"
            )
            avg_recons = 0.0

    script_args.eval_iter = script_args.eval_iter * 10
    fid_his, l2_his = [], []
    avg_recons = 0.0
    train_start_time = time.time()
    for nb_iter in range(start_iter + 1, script_args.total_iter + 1):
        gt_full = next(train_loader_iter).to(device).float()
        gt_motion = gt_full[..., full_mask]
        latents = tokenizer(gt_full)
        pred_motion = decoder(latents)
        loss_motion = loss_fn(pred_motion, gt_motion)
        loss = loss_motion

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        avg_recons += loss_motion.item()

        if nb_iter % script_args.print_iter == 0:
            avg_recons /= script_args.print_iter
            writer.add_scalar("./Train/L1", avg_recons, nb_iter)
            elapsed = time.time() - train_start_time
            completed_since_resume = max(1, nb_iter - start_iter)
            iter_time = elapsed / completed_since_resume
            eta = iter_time * max(0, script_args.total_iter - nb_iter)
            progress = 100.0 * nb_iter / script_args.total_iter
            logger.info(f"Train. Iter {nb_iter}/{script_args.total_iter} ({progress:.2f}%): Recons. {avg_recons:.5f} \t {iter_time:.3f}s/it \t elapsed {format_duration(elapsed)} \t eta {format_duration(eta)}")
            avg_recons = 0.0

        if nb_iter % script_args.eval_iter == 0:
            logger.info(f"Starting eval/checkpoint at iter {nb_iter}; eval interval is every {script_args.eval_iter} train iterations")
            save_checkpoint(decoder, optimizer, scheduler, data_args, script_args, full_mask, out_dir, nb_iter)
            fid, l2 = evaluate(
                decoder,
                tokenizer,
                test_loader,
                eval_copy,
                mean_pose,
                std_pose,
                full_mask,
                out_dir,
                nb_iter,
                fid_his,
                l2_his,
                device,
                logger,
            )
            writer.add_scalar("./Eval/FID", fid, nb_iter)
            writer.add_scalar("./Eval/L2", l2, nb_iter)


if __name__ == "__main__":
    main()
