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

    @staticmethod
    def _guidance_iteration_counts(t, control, batch_size: int, device) -> torch.Tensor:
        override = control.get("n_iters")
        if override is not None:
            counts = torch.as_tensor(override, device=device, dtype=torch.long)
            if counts.dim() == 0:
                counts = counts.expand(batch_size)
            return counts.reshape(-1)[:batch_size].clamp_min(0)

        if not torch.is_tensor(t):
            t_tensor = torch.as_tensor(t, device=device)
        else:
            t_tensor = t.to(device=device)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand(batch_size)
        t_tensor = t_tensor.reshape(-1)[:batch_size].to(dtype=torch.float32)

        late_start = int(control.get("late_start", 300))
        iters_early = int(control.get("iters_early", 5))
        iters_late = int(control.get("iters_late", 30))
        if late_start <= 0:
            progress = (t_tensor <= late_start).to(dtype=torch.float32)
        else:
            progress = ((late_start - t_tensor) / float(late_start)).clamp(0.0, 1.0)
        counts = torch.round(
            iters_early + (iters_late - iters_early) * progress
        ).to(dtype=torch.long)
        return counts.clamp_min(0)

    def _spatial_guidance_step(self, x0, t, guidance_fn, control):
        hint = control["hint"].to(device=x0.device, dtype=x0.dtype)
        mask = control["mask"].to(device=x0.device, dtype=x0.dtype)
        if mask.sum() <= 0:
            return x0

        if hint.dim() == 3:
            hint = hint.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        n_iters_by_sample = self._guidance_iteration_counts(
            t, control, x0.shape[0], x0.device
        )
        n_iters = int(n_iters_by_sample.max().item())
        if n_iters <= 0:
            return x0

        active = torch.clamp((mask > 0).sum().to(dtype=x0.dtype), min=1.0)
        active_norm = str(control.get("active_norm", "linear")).strip().lower()
        if active_norm == "linear":
            norm = active.item()
        elif active_norm == "sqrt":
            norm = active.sqrt().item()
        elif active_norm in ("none", "off"):
            norm = 1.0
        else:
            raise ValueError(f"Unknown guidance active_norm: {active_norm}")
        lr = float(control.get("scale", 20.0)) / max(norm, 1.0)
        weight = float(control.get("weight", 1.0))
        log_every = int(control.get("log_every", 0))
        loss_stop = float(control.get("loss_stop", 5e-7))
        log_context = control.get("log_context", "")
        with torch.enable_grad():
            x = x0.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=lr)
            for opt_i in range(n_iters):
                iter_active = (n_iters_by_sample > opt_i).to(dtype=x0.dtype)
                if iter_active.sum() <= 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                guidance_kwargs = control.get("guidance_kwargs", {})
                joints = guidance_fn(
                    x,
                    bool(control.get("freeze_root", True)),
                    **guidance_kwargs,
                )
                frame_mask = mask.unsqueeze(-1) * iter_active.view(-1, 1, 1, 1)
                diff = (joints - hint) * frame_mask
                loss = 0.5 * diff.pow(2).sum() * weight
                loss.backward()
                grad_norm = x.grad.detach().norm().item() if x.grad is not None else 0.0
                optimizer.step()
                loss_value = loss.detach().item()
                if log_every > 0 and (opt_i == 0 or opt_i == n_iters - 1 or (opt_i + 1) % log_every == 0):
                    context = f" {log_context}" if log_context else ""
                    if torch.is_tensor(t):
                        t_log = t.detach().cpu().tolist() if t.dim() > 0 else int(t.item())
                    else:
                        t_log = int(t)
                    print(
                        f"[guidance]{context} t={t_log} opt={opt_i + 1}/{n_iters} "
                        f"loss={loss_value:.6f} grad={grad_norm:.6f} lr={lr:.6f} active_norm={active_norm} "
                        f"n_iters={n_iters_by_sample.detach().cpu().tolist()}"
                    )
                if loss_value <= loss_stop:
                    break
        return x.detach()

    @staticmethod
    def _slice_control(control, indices):
        sliced = dict(control)
        for key in ("hint", "mask"):
            if key in sliced:
                sliced[key] = sliced[key].index_select(0, indices)
        return sliced

    def _diffusion_reverse_delayed(
            self,
            latents: torch.Tensor,
            seed: torch.Tensor,
            at_feat: torch.Tensor,
            timesteps: torch.Tensor,
            noise: torch.Tensor,
            guidance_scale: float,
            control: dict,
            guidance_fn,
            seed_update_fn,
            extra_step_kwargs: dict,
            chunk_delay: int,
    ) -> torch.Tensor:
        num_steps = len(timesteps)
        num_chunks = latents.shape[0]
        post_iters = int(control.get("post_iters", 0)) if control is not None else 0
        post_wave_steps = post_iters if guidance_fn is not None else 0
        total_waves = num_steps + post_wave_steps + (num_chunks - 1) * chunk_delay

        latents = torch.zeros_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])
        x0_estimate = latents.detach().clone()

        logger.info(
            "[delay] enabled chunks=%s ddim_steps=%s chunk_delay=%s post_waves=%s total_waves=%s",
            num_chunks, num_steps, chunk_delay, post_wave_steps, total_waves,
        )

        first_chunk_timing = {
            "first_chunk_seconds": 0.0,
            "first_chunk_waves": num_steps + post_wave_steps,
            "first_chunk_diffusion_steps": num_steps,
            "first_chunk_post_steps": post_wave_steps,
        }
        first_chunk_start_time = None
        first_chunk_end_local = num_steps + post_wave_steps - 1

        chunk_ids = torch.arange(num_chunks, device=latents.device)
        for wave_i in range(total_waves):
            if wave_i == 0:
                if latents.is_cuda:
                    torch.cuda.synchronize(latents.device)
                first_chunk_start_time = time.perf_counter()
            local_steps = wave_i - chunk_ids * chunk_delay
            diffusion_mask = (local_steps >= 0) & (local_steps < num_steps)
            post_mask = (local_steps >= num_steps) & (local_steps < num_steps + post_wave_steps)
            active_mask = diffusion_mask | post_mask
            if not active_mask.any():
                continue

            diffusion_indices = diffusion_mask.nonzero(as_tuple=False).flatten()
            active_indices = active_mask.nonzero(as_tuple=False).flatten()
            active_t = timesteps.new_zeros((active_indices.shape[0],))
            candidate_x0 = latents.index_select(0, active_indices)

            model_output = None
            active_latents = None
            active_steps = None
            guided_x0 = None

            if diffusion_indices.numel() > 0:
                active_steps = local_steps.index_select(0, diffusion_indices).long()
                diffusion_t = timesteps.index_select(0, active_steps)

                if seed_update_fn is not None:
                    updated_seed = seed_update_fn(x0_estimate.detach(), seed).detach()
                    seed = seed.clone()
                    seed[diffusion_indices] = updated_seed[diffusion_indices]

                active_latents = latents.index_select(0, diffusion_indices)
                active_input = torch.cat([
                    self.scheduler.scale_model_input(
                        active_latents[local_i:local_i + 1],
                        diffusion_t[local_i],
                    )
                    for local_i in range(active_latents.shape[0])
                ], dim=0)
                model_output = self.apply_classifier_free_guidance(
                    x=active_input,
                    timesteps=diffusion_t,
                    seed=seed.index_select(0, diffusion_indices),
                    at_feat=at_feat.index_select(0, diffusion_indices),
                    guidance_scale=guidance_scale,
                )

                latents_pred_x0, _ = self.predicted_origin(
                    model_output, diffusion_t, active_latents
                )
                diffusion_pos = torch.searchsorted(active_indices, diffusion_indices)
                candidate_x0 = candidate_x0.clone()
                candidate_x0[diffusion_pos] = latents_pred_x0
                active_t[diffusion_pos] = diffusion_t

            if guidance_fn is not None:
                step_control = self._slice_control(control, active_indices)
                n_iters_by_active = torch.zeros(
                    active_indices.shape[0], device=latents.device, dtype=torch.long
                )
                if diffusion_indices.numel() > 0:
                    diffusion_pos = torch.searchsorted(active_indices, diffusion_indices)
                    n_iters_by_active[diffusion_pos] = self._guidance_iteration_counts(
                        active_t.index_select(0, diffusion_pos),
                        control,
                        diffusion_pos.shape[0],
                        latents.device,
                    )
                if post_wave_steps > 0:
                    post_indices = post_mask.nonzero(as_tuple=False).flatten()
                    if post_indices.numel() > 0:
                        post_pos = torch.searchsorted(active_indices, post_indices)
                        n_iters_by_active[post_pos] = 1
                step_control["n_iters"] = n_iters_by_active
                step_control["log_context"] = (
                    f"wave={wave_i:02d} chunks={active_indices.detach().cpu().tolist()} "
                    f"local={local_steps.index_select(0, active_indices).detach().cpu().tolist()}"
                )
                x0_context = x0_estimate.clone()
                x0_context[active_indices] = candidate_x0.detach()
                step_control["guidance_kwargs"] = {
                    "full_x0": x0_context,
                    "chunk_indices": active_indices,
                }
                guided_x0 = self._spatial_guidance_step(
                    candidate_x0,
                    active_t,
                    guidance_fn,
                    step_control,
                )
            else:
                guided_x0 = candidate_x0

            latents = latents.clone()
            x0_estimate = x0_estimate.clone()

            if diffusion_indices.numel() > 0:
                diffusion_pos = torch.searchsorted(active_indices, diffusion_indices)
                guided_diffusion_x0 = guided_x0.index_select(0, diffusion_pos)
                diffusion_t = active_t.index_select(0, diffusion_pos)
                if guidance_fn is not None:
                    model_output = self.model_output_from_origin(
                        guided_diffusion_x0, diffusion_t, active_latents
                    )
                next_latents = []
                for local_i in range(diffusion_indices.shape[0]):
                    next_latents.append(
                        self.scheduler.step(
                            model_output[local_i:local_i + 1],
                            diffusion_t[local_i],
                            active_latents[local_i:local_i + 1],
                            **extra_step_kwargs,
                        ).prev_sample
                    )
                next_latents = torch.cat(next_latents, dim=0)
                latents[diffusion_indices] = next_latents
                x0_estimate[diffusion_indices] = guided_diffusion_x0.detach()

                finished_mask = active_steps == (num_steps - 1)
                if finished_mask.any():
                    finished_indices = diffusion_indices.index_select(
                        0, finished_mask.nonzero(as_tuple=False).flatten()
                    )
                    finished_latents = next_latents.index_select(
                        0, finished_mask.nonzero(as_tuple=False).flatten()
                    )
                    x0_estimate[finished_indices] = finished_latents.detach()

            if post_wave_steps > 0:
                post_indices = post_mask.nonzero(as_tuple=False).flatten()
                if post_indices.numel() > 0:
                    post_pos = torch.searchsorted(active_indices, post_indices)
                    guided_post = guided_x0.index_select(0, post_pos)
                    latents[post_indices] = guided_post
                    x0_estimate[post_indices] = guided_post.detach()

            if (
                    first_chunk_start_time is not None
                    and first_chunk_timing["first_chunk_seconds"] <= 0.0
                    and int(local_steps[0].item()) == first_chunk_end_local
            ):
                if latents.is_cuda:
                    torch.cuda.synchronize(latents.device)
                first_chunk_timing["first_chunk_seconds"] = time.perf_counter() - first_chunk_start_time

        return latents, first_chunk_timing




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

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        delayed_timing = {}
        if chunk_delay > 0:
            latents, delayed_timing = self._diffusion_reverse_delayed(
                latents,
                seed,
                at_feat,
                timesteps,
                noise,
                guidance_scale,
                control,
                guidance_fn,
                seed_update_fn,
                extra_step_kwargs,
                chunk_delay,
            )
        else:
            latents = torch.zeros_like(latents)
            latents = self.scheduler.add_noise(latents, noise, timesteps[0])

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

        post_iters = int(control.get("post_iters", 0)) if control is not None else 0
        if chunk_delay == 0 and post_iters > 0 and guidance_fn is not None:
            post_control = dict(control)
            post_control["iters_early"] = post_iters
            post_control["iters_late"] = post_iters
            post_control["late_start"] = 0
            post_control.pop("guidance_kwargs", None)
            post_control["log_context"] = (
                f"post_diffusion chunks={list(range(latents.shape[0]))}"
            )
            post_t = timesteps.new_zeros(())
            latents = self._spatial_guidance_step(
                latents,
                post_t,
                guidance_fn,
                post_control,
            )

        return_dict['latents'] = latents
        return_dict['timing'] = {
            key: torch.as_tensor(value, device=latents.device)
            for key, value in delayed_timing.items()
        }
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
