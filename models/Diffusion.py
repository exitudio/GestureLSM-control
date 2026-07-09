import time
import inspect
import logging
from typing import Optional
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters, extract_into_tensor, sum_flat

logger = logging.getLogger(__name__)


class GestureDiffusion(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')

        self.seq_len = self.denoiser.seq_len
        self.input_dim = self.denoiser.input_dim
        self.num_joints = self.denoiser.joint_num

    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Scheduler: {count_parameters(self.modality_encoder)}M')

    def apply_classifier_free_guidance(self, x, timesteps, seed, at_feat, guidance_scale=1.0):
        """
        Apply classifier-free guidance by running both conditional and unconditional predictions.
        
        Args:
            x: Input tensor
            timesteps: Timestep tensor
            seed: Seed vectors
            at_feat: Audio features
            guidance_scale: Guidance scale (1.0 means no guidance)
            
        Returns:
            Guided output tensor
        """
        batch_size = x.shape[0]
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.shape[0] != batch_size:
            raise ValueError(
                f"timesteps batch mismatch: got {timesteps.shape[0]}, expected {batch_size}"
            )

        if guidance_scale <= 1.0:
            # No guidance needed, run normal forward pass
            return self.denoiser(
                x=x,
                timesteps=timesteps,
                seed=seed,
                at_feat=at_feat,
                cond_drop_prob=0.0,
                null_cond=False
            )
        
        # Double the batch for classifier free guidance
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([seed] * 2, dim=0)
        at_feat_doubled = torch.cat([at_feat] * 2, dim=0)
        timesteps_doubled = torch.cat([timesteps, timesteps], dim=0)
        
        # Create conditional and unconditional audio features
        batch_size = at_feat.shape[0]
        null_cond_embed = self.denoiser.null_cond_embed.to(at_feat.dtype)
        at_feat_uncond = null_cond_embed.unsqueeze(0).expand(batch_size, -1, -1)
        at_feat_combined = torch.cat([at_feat, at_feat_uncond], dim=0)
        
        # Run both conditional and unconditional predictions
        output = self.denoiser(
            x=x_doubled,
            timesteps=timesteps_doubled,
            seed=seed_doubled,
            at_feat=at_feat_combined,
        )
        
        # Split predictions and apply guidance
        pred_cond, pred_uncond = output.chunk(2, dim=0)
        guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        return guided_output

    def apply_conditional_dropout(self, at_feat, cond_drop_prob=0.1):
        """
        Apply conditional dropout during training to simulate classifier-free guidance.
        
        Args:
            at_feat: Audio features tensor
            cond_drop_prob: Probability of dropping conditions (default 0.1)
            
        Returns:
            Modified audio features with some conditions replaced by null embeddings
        """
        batch_size = at_feat.shape[0]
        
        # Create dropout mask
        keep_mask = torch.rand(batch_size, device=at_feat.device) > cond_drop_prob
        
        # Create null condition embeddings
        null_cond_embed = self.denoiser.null_cond_embed.to(at_feat.dtype)
        
        # Apply dropout: replace dropped conditions with null embeddings
        at_feat_dropped = at_feat.clone()
        at_feat_dropped[~keep_mask] = null_cond_embed.unsqueeze(0).expand((~keep_mask).sum(), -1, -1)
        
        return at_feat_dropped

    def predicted_origin(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> tuple:
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        # i will do this
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output

        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas

        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alphas * sample - sigmas * model_output
            pred_epsilon = alphas * model_output + sigmas * sample
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

        return pred_original_sample, pred_epsilon

    def model_output_from_origin(self, pred_original_sample: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        self.alphas = self.alphas.to(pred_original_sample.device)
        self.sigmas = self.sigmas.to(pred_original_sample.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        if self.scheduler.config.prediction_type == "epsilon":
            return (sample - alphas * pred_original_sample) / sigmas
        if self.scheduler.config.prediction_type == "sample":
            return pred_original_sample
        if self.scheduler.config.prediction_type == "v_prediction":
            pred_epsilon = (sample - alphas * pred_original_sample) / sigmas
            return alphas * pred_epsilon - sigmas * pred_original_sample
        raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

    def _guidance_variance(self, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        betas = self.scheduler.betas.to(sample.device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(sample.device)
        alphas_cumprod_prev = torch.cat([
            torch.ones_like(alphas_cumprod[:1]), alphas_cumprod[:-1],
        ])
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_variance = torch.cat([
            posterior_variance[1:2], posterior_variance[1:],
        ])
        return extract_into_tensor(posterior_variance, timesteps, sample.shape)

    def _spatial_guidance_step(self, x0, t, guidance_fn, control):
        hint = control["hint"].to(device=x0.device, dtype=x0.dtype)
        mask = control["mask"].to(device=x0.device, dtype=x0.dtype)
        if mask.sum() <= 0:
            return x0

        if hint.dim() == 3:
            hint = hint.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        late_start = int(control.get("late_start", 300))
        t_for_schedule = t.min() if t.dim() > 0 else t
        n_iters = int(control.get("iters_late", 30) if int(t_for_schedule.item()) <= late_start else control.get("iters_early", 5))
        if n_iters <= 0:
            return x0

        active = torch.clamp((mask > 0).sum().to(dtype=x0.dtype), min=1.0)
        lr = float(control.get("scale", 20.0)) / active.item()
        weight = float(control.get("weight", 1.0))
        log_every = int(control.get("log_every", 0))
        loss_stop = float(control.get("loss_stop", 5e-7))
        log_context = control.get("log_context", "")
        with torch.enable_grad():
            x = x0.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=lr)
            for opt_i in range(n_iters):
                optimizer.zero_grad(set_to_none=True)
                joints = guidance_fn(x, bool(control.get("freeze_root", True)))
                frame_mask = mask.unsqueeze(-1)
                diff = (joints - hint) * frame_mask
                loss = 0.5 * diff.pow(2).sum() * weight
                loss.backward()
                grad_norm = x.grad.detach().norm().item() if x.grad is not None else 0.0
                optimizer.step()
                loss_value = loss.detach().item()
                if log_every > 0 and (opt_i == 0 or opt_i == n_iters - 1 or (opt_i + 1) % log_every == 0):
                    context = f" {log_context}" if log_context else ""
                    t_log = t.detach().cpu().tolist() if t.dim() > 0 else int(t.item())
                    print(
                        f"[guidance]{context} t={t_log} opt={opt_i + 1}/{n_iters} "
                        f"loss={loss_value:.6f} grad={grad_norm:.6f} lr={lr:.6f}"
                    )
                if loss_value <= loss_stop:
                    break
        return x.detach()



    def forward(self, cond_: dict) -> dict:

        audio = cond_['y']['audio_onset']
        word = cond_['y']['word']
        id = cond_['y']['id']
        seed = cond_['y']['seed']
        style_feature = cond_['y']['style_feature']
        control = cond_['y'].get('control')
        guidance_fn = cond_['y'].get('guidance_fn')
        seed_update_fn = cond_['y'].get('seed_update_fn')

        audio_feat = self.modality_encoder(audio, word)

        bs = audio_feat.shape[0]
        shape_ = (bs, self.input_dim * self.num_joints, 1, self.seq_len)
        latents = torch.randn(shape_, device=audio_feat.device)

        latents = self._diffusion_reverse(
            latents, seed, audio_feat, guidance_scale=self.guidance_scale,
            control=control, guidance_fn=guidance_fn,
            seed_update_fn=seed_update_fn)

        return latents



    def _diffusion_reverse(
            self,
            latents: torch.Tensor,
            seed: torch.Tensor,
            at_feat: torch.Tensor,
            guidance_scale: float = 1,
            control: Optional[dict] = None,
            guidance_fn = None,
            seed_update_fn = None,
    ) -> torch.Tensor:

        return_dict = {}
        # scale the initial noise by the standard deviation required by the scheduler, like in Stable Diffusion
        # this is the initial noise need to be returned for rectified training
        latents = latents * self.scheduler.init_noise_sigma
        noise = latents

        return_dict["init_noise"] = latents
        return_dict['at_feat'] = at_feat
        return_dict['seed'] = seed

        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(at_feat.device)
        num_steps = len(timesteps)
        chunk_delay = max(0, int(control.get("chunk_delay", 0))) if control is not None else 0
        if chunk_delay > 0:
            logger.warning(
                "Ignoring chunk_delay=%s; delayed wave scheduling is disabled.",
                chunk_delay,
            )

        latents = torch.zeros_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        logger.info(
            "[delay] disabled chunks=%s ddim_steps=%s",
            latents.shape[0], num_steps,
        )
        for wave_i, t in enumerate(timesteps):
            latent_model_input = latents
            # actually it does nothing here according to ddim scheduler
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            model_output = self.apply_classifier_free_guidance(
                x=latent_model_input,
                timesteps=t,
                seed=seed,
                at_feat=at_feat,
                guidance_scale=guidance_scale)

            t_batch = None
            x0_for_seed = None
            if control is not None and guidance_fn is not None:
                t_batch = t.expand(latents.shape[0])
                latents_pred_x0, _ = self.predicted_origin(model_output, t_batch, latents)
                step_control = dict(control)
                step_control["log_context"] = (
                    f"wave={wave_i:02d} chunks={list(range(latents.shape[0]))}"
                )
                guided_x0 = self._spatial_guidance_step(
                    latents_pred_x0, t, guidance_fn, step_control
                )
                model_output = self.model_output_from_origin(guided_x0, t_batch, latents)
                x0_for_seed = guided_x0

            if seed_update_fn is not None:
                if x0_for_seed is None:
                    if t_batch is None:
                        t_batch = t.expand(latents.shape[0])
                    x0_for_seed, _ = self.predicted_origin(model_output, t_batch, latents)
                seed = seed_update_fn(x0_for_seed.detach(), seed).detach()

            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample

        return_dict['latents'] = latents
        return return_dict

    def _diffusion_process(self,
            latents: torch.Tensor,
            audio_feat: torch.Tensor,
            id: torch.Tensor,
            seed: torch.Tensor,
            style_feature: torch.Tensor
        ) -> dict:

        # [batch_size, n_frame, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]


        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        )

        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        model_output = self.denoiser(
            x=noisy_latents,
            timesteps=timesteps,
            seed=seed,
            at_feat=audio_feat,
        )

        latents_pred, noise_pred = self.predicted_origin(model_output, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
            "timesteps": timesteps,
            "model_output": model_output,
        }
        return n_set

    def train_forward(self, cond_: dict, x0: torch.Tensor) -> dict:
        audio = cond_['y']['audio_onset']
        word = cond_['y']['word']
        id = cond_['y']['id']
        seed = cond_['y']['seed']
        style_feature = cond_['y']['style_feature']

        audio_feat = self.modality_encoder(audio, word)
        
        # Apply conditional dropout during training
        audio_feat = self.apply_conditional_dropout(audio_feat, cond_drop_prob=0.1)
        
        n_set = self._diffusion_process(x0, audio_feat, id, seed, style_feature)

        loss_dict = dict()

        # Diffusion loss
        if self.scheduler.config.prediction_type == "epsilon":
            model_pred, target = n_set['noise_pred'], n_set['noise']
        elif self.scheduler.config.prediction_type == "sample":
            model_pred, target = n_set['sample_pred'], n_set['sample_gt']
        elif self.scheduler.config.prediction_type == "v_prediction":
            # For v_prediction, we need to compute the v target
            # v = alpha * noise - sigma * x0
            timesteps = n_set['timesteps']

            self.alphas = self.alphas.to(x0.device)
            self.sigmas = self.sigmas.to(x0.device)
            alphas = extract_into_tensor(self.alphas, timesteps, x0.shape)
            sigmas = extract_into_tensor(self.sigmas, timesteps, x0.shape)

            v_target = alphas * n_set['noise'] - sigmas * n_set['sample_gt']
            model_pred, target = n_set['model_output'], v_target  # The model output is the v prediction
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")


        # mse loss
        diff_loss = F.mse_loss(target, model_pred, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        return loss_dict
