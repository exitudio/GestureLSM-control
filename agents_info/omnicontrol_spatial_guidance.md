# Spatial Guidance Integration: OmniControl → GestureLSM (Inference-only, DDIM)

## Context

GestureLSM generates co-speech gestures from audio using a latent-space diffusion
model. Today the generation is conditioned only on audio and word features — there
is no way to ask the model "place the left wrist at position (x, y, z) at frame
100" or any similar spatial constraint.

OmniControl (Xie et al., ICLR 2024, arXiv 2310.08580; code at
`~/git/OmniControl`) solves this for the HumanML3D / MDM family with two
mechanisms. The user wants only the first — **training-free spatial guidance at
inference** — adopted in this repo:

> At each denoising step, take the model's predicted clean motion (the posterior
> mean `μ` of `p(x_{t-1} | x_t)`), compute joint positions from it, evaluate a
> loss against target joint positions, and run gradient descent **on `μ`** to
> drive it toward satisfying the constraint. The denoiser is **not** in this
> inner loop.

The intended outcome is a single-binary inference path (extended `demo_html.py`)
that accepts an optional sparse joint-position control hint and produces a
generation that approximately satisfies the hint while remaining audio-driven.

### Decisions locked in with the user

- **Sampler / checkpoint**: target the **DDIM path** — `GestureDiffusion` in
  `models/Diffusion.py`, config `configs/diffuser_rvqvae_128.yaml`, checkpoint
  `ckpt/new_540_diffusion.bin`. Many steps means usable guidance headroom (the
  shortcut/flow-matching path has too few steps for iterative guidance and is
  out of scope for this plan).
- **First scenario**: **multi-joint sparse keyframes** — several joints at
  several frames. The control format must support this from day 1.
- **Entrypoint**: extend the existing `demo_html.py` with an opt-in
  `--control_json` flag (no new top-level script).
- **Renderer**: keep the existing **2-skeleton view** (controlled generation vs
  GT). Do not add a third "uncontrolled" skeleton. `write_pose_html` unchanged.

## How GestureLSM handles long sequences (chunked rolling window)

Unlike OmniControl/MDM (which denoise an entire ≤196-frame motion in a single
DDIM pass), GestureLSM uses **chunked sequential generation** because the
denoiser has a fixed sequence length. This is fundamental to how guidance must
be wired.

### Sizes

| Parameter | Value | Meaning |
|---|---|---|
| `pose_length` | 128 motion frames | One chunk seen by the denoiser at a time (≈ 4.27 s @ 30 fps) |
| `vqvae_squeeze_scale` | 4 | RVQVAE compresses 4 motion frames → 1 latent timestep |
| `denoiser.seq_len` | 32 latent timesteps | = `pose_length / vqvae_squeeze_scale` (fixed by the model: `models/denoiser.py:53`, also the size of `null_cond_embed` at `denoiser.py:71`) |
| `pre_frames` | 4 latent timesteps = 16 motion frames | Overlap between consecutive chunks (the "seed") |
| `round_l` | `pose_length - pre_frames × vqvae_squeeze_scale` = 112 motion frames | Net new motion frames produced per chunk |

### How a long sequence is produced (`demo_html.HTMLTrainer._g_test`)

For a 60-second audio (1808 frames), the trainer runs ~16 sequential chunks.
Each chunk:

1. Slices the chunk's audio and word features.
2. Builds a `seed`: for chunk 0 it's the initial seed latent; for chunk
   `i > 0` it's the **last `pre_frames` latent steps of the previously
   generated chunk** (`last_sample[:, -pre_frames:, :]`).
3. Runs **a full DDIM sampling loop** on a 32-latent-step window, conditioned
   on the audio chunk + seed.
4. Drops the first `pre_frames` latent steps of the chunk's output (those
   correspond to the seed/overlap region) and appends the remaining
   `seq_len - pre_frames = 28` latent steps to the result.

Chunks are produced **sequentially** — chunk `i+1` cannot start until chunk
`i` finishes, because it consumes chunk `i`'s tail as its seed.

### Consequences for spatial guidance

1. **Guidance is per-chunk**, not global. A keyframe at motion frame `f`
   belongs to chunk `i ≈ f // 112`, and can only influence the DDIM sampling
   of that one chunk.
2. **Causality is one-way**: a keyframe in chunk 5 cannot influence chunks
   0–4 (they were already denoised and the seeds for them are gone). But a
   keyframe in chunk 5 *will* affect chunks ≥6 indirectly, because chunk 5's
   guided tail becomes chunk 6's seed (so the wrist position carries forward
   naturally — usually a good thing for continuity).
3. **Hint/mask must be sliced per chunk** at the same boundaries the trainer
   uses. Frames in the overlap region (last 16 of chunk `i` = first 16 of
   chunk `i+1` after the seed is consumed) only need to be active in chunk
   `i`; chunk `i+1` already inherits those frames via the seed and shouldn't
   re-guide them.
4. **Per-chunk inference cost multiplies**. If we pick `n_iters = 30` for the
   late timesteps, that's 30 × (RVQVAE decode + SMPLX backward) **per chunk**,
   × (number of guided outer steps), × (number of chunks). A 60-second clip
   will cost roughly 16× a single-chunk inference. Plan for this when
   choosing schedules.

## What OmniControl actually does (verified against `~/git/OmniControl`)

The core implementation lives in `diffusion/gaussian_diffusion.py`. The hook is
called inside `p_sample` (line 535-537):

```python
out = self.p_mean_variance(model, x, t, ...)
if 'hint' in model_kwargs['y'].keys():
    out['mean'] = self.guide(out['mean'], t, model_kwargs=model_kwargs)
# then x_{t-1} = out['mean'] + noise * sqrt(variance)
```

`self.guide` (line 450) does **gradient descent on the posterior mean**:

```python
for _ in range(n_guide_steps):
    loss, grad = self.gradients(x, hint, mask_hint, joint_ids)
    grad = model_variance * grad           # classifier guidance scaling
    if t[0] >= t_stopgrad:
        x = x - scale * grad
```

And `self.gradients` (line 423):

```python
with torch.enable_grad():
    x.requires_grad_(True)
    x_ = x.permute(0, 3, 2, 1).contiguous().squeeze(2)
    x_ = x_ * self.std + self.mean           # de-normalize the rep
    joint_pos = recover_from_ric(x_, 22)      # deterministic FK (no NN)
    loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
    grad = torch.autograd.grad([loss.sum()], [x])[0]
    grad[..., 0] = 0                          # zero the root joint
```

The five details that matter most for porting:

1. **Guidance is applied to the posterior mean `μ`, not to `x_t`**, and the
   denoiser is run **only once per outer step** (to produce `μ`). The inner
   loop only re-runs the forward kinematics. This is what makes large
   `n_guide_steps` affordable.
2. **`n_guide_steps` schedule** (`gaussian_diffusion.py:467-470`): inference
   uses **10 iterations** for `t ≥ 10` and **500 iterations** for `t < 10`.
   Big push at the end of denoising where the predicted clean estimate is
   accurate.
3. **Adaptive scale** (`gaussian_diffusion.py:443-448`):
   `scale = 20 / max_keyframes_in_batch`. Fewer keyframes → bigger per-step
   nudge per active joint.
4. **Variance-weighted gradient** (`gaussian_diffusion.py:493`):
   `grad = model_variance * grad`. Standard DDPM classifier guidance form.
5. **Root-joint gradient is zeroed** (`gaussian_diffusion.py:439`) because
   HumanML3D motions start at the origin and the model has no notion of
   absolute world position. For SMPLX we need an analogous choice — see
   "Open items" below.

Note the cost asymmetry: OmniControl's per-iteration cost is essentially the
cost of `recover_from_ric`, a deterministic quaternion-rotation +
indexing routine (`data_loaders/humanml/scripts/motion_process.py:415`). No
neural network is invoked. Our per-iteration cost will be 3 × RVQVAE decoder +
6D→axis-angle → SMPLX LBS, plus the backward, which is heavier.

## Critical files to read / modify

Read for context (no changes):

- `models/Diffusion.py` — the DDIM sampler. The inference loop is
  `GestureDiffusion._diffusion_reverse` (≈ line 162); the per-step
  `scheduler.step` call (≈ line 211) is where guidance must be inserted.
  `apply_classifier_free_guidance` (≈ line 38) is reusable.
- `models/Diffusion.py:245` — there is already a `predicted_origin(...)`
  helper that converts denoiser output into the predicted clean latent given
  the schedule alphas/sigmas. We reuse this to obtain `μ`-equivalent for
  guidance.
- `models/vq/model.py:102` — `latent2origin`. **For un-guided inference it
  re-quantizes (Gumbel sampling at `sample_codebook_temp=0.5`) and then
  decodes.** For the guidance path we do **not** want to go through the
  quantizer: (1) the diffusion model is already trained on the pre-quant
  continuous latents (`map2latent` returns the encoder output without
  quantization, see `model.py:95`), so the continuous prediction is itself a
  valid input to the decoder; (2) avoiding the quantizer removes the
  stochasticity from Gumbel sampling, gives a deterministic forward, and the
  gradient is exact (no straight-through estimator needed).
  → Add a thin `RVQVAE.decode_continuous(x)` helper that mirrors
  `latent2origin` minus the quantizer call (permute, run `self.decoder(x)`).
  Use it in the guidance path. Keep `latent2origin` unchanged for the
  un-guided demo path.
- `models/vq/quantizer.py`, `models/vq/residual_vq.py` — read for context
  only; not modified. They contain STE / pass-through, which we are
  side-stepping by skipping the quantizer entirely.
- `demo_html.py` — current single-binary inference entry. The trainer
  (`HTMLTrainer`) already loads the per-body-part RVQVAE decoders
  (`self.vq_model_upper / _hands / _lower`), the `mean_*` / `std_*` /
  `trans_mean` / `trans_std` normalizers, the SMPLX model, the joint masks,
  and uses `utils/rotation_conversions` for the 6D→matrix→axis-angle chain.
  All of these are reusable in the differentiable decode helper.
- `dataloaders/beat_sep_single.py:CustomDataset` — confirms the audio /
  textgrid loading path is unchanged.
- `configs/diffuser_rvqvae_128.yaml`, `configs/model_config.yaml` — DDIM
  configuration. Currently `g_name: GestureDiffusion`. Default audio + scheduler
  parameters live here.

Files to modify:

- `models/vq/model.py` — add `RVQVAE.decode_continuous(x)`: mirrors
  `latent2origin` minus the quantizer; permute the input and run
  `self.decoder` directly. Keep `latent2origin` unchanged.
- `models/Diffusion.py`:
  - Extend `GestureDiffusion.forward` / `_diffusion_reverse` to accept an
    optional `control` dict and `guidance_fn` closure.
  - Add `GestureDiffusion._spatial_guidance_step(mu, t, guidance_fn, control,
    lr_base, num_iters)`. This is the analog of OmniControl's `guide()`.
  - Add a small helper to compute posterior `mean` (and `variance` if we want
    classifier-style scaling) from `predicted_origin` and `x_t` at timestep `t`
    using `self.scheduler` alphas/sigmas. For DDIM (deterministic) this
    collapses to "the predicted clean latent" — we'll guide that directly.
- `demo_html.py`:
  - Add CLI flag `--control_json` pointing to a JSON file with entries like
    `{"frame": int, "joint": int, "xyz": [x, y, z]}` (and optional `weight`).
  - Load the JSON in `main`, build dense `hint [N, 55, 3]` and
    `mask [N, 55]` tensors, pack into a `control` dict
    (`hint`, `mask`, `weight`).
  - Build the differentiable `latent → joints` closure (Phase 1 below) bound
    to the trainer's already-loaded decoders / normalizers / SMPLX / joint
    masks.
  - Pass `control` + `guidance_fn` + hyperparameters into the model. When
    `--control_json` is omitted, behaviour must be byte-identical to today's
    `demo_html.py`.
  - Render `generation.html` exactly as today: controlled generation vs GT
    (no third skeleton).
- Note for the user: `demo_html.py` currently defaults to the **LSM** config
  (`shortcut_rvqvae_128_hf.yaml`). To exercise this plan they must run with
  `--config configs/diffuser_rvqvae_128.yaml`. Document this in the script's
  help string; do not change the default.

No new top-level script; no realism module; no training.

## Recommended approach

### Phase 1 — Differentiable `latent → joints` helper

Implemented in `demo_html.py` (or a small `models/guidance.py` if it grows;
prefer keeping it in `demo_html.py` to start). The helper takes a continuous
latent of shape `[B, 384, 1, T_latent]` (= 3 × 128 channels) and a reference to
the trainer, and returns SMPLX joints `[B, T_motion, 55, 3]`.

The implementation reuses, in order:

1. The same latent split (upper / hands / lower) and `vqvae_latent_scale`
   multiplication done in `HTMLTrainer._g_test`.
2. **`decode_continuous` per body part** (new helper on `RVQVAE`,
   `models/vq/model.py`). Skips the quantizer; runs the decoder directly on
   the continuous diffusion-predicted latent. Deterministic, exact gradient,
   no STE.
3. The trainer's `mean_upper / std_upper / mean_hands / std_hands /
   mean_lower / std_lower` for de-normalization.
4. The existing translation handling (`trans_mean / trans_std` plus the
   `cumsum` with `Y` overwrite). Differentiable.
5. `utils/rotation_conversions` for 6D → matrix → axis-angle.
6. The trainer's `inverse_selection_tensor` and joint masks
   (`joint_mask_upper / _hands / _lower`) to reassemble the 165-d pose.
7. The trainer's SMPLX model — reuse the call pattern from
   `demo_html.smplx_to_joints`.

Unit test: random latent, sum joints, `.backward()`, assert `latent.grad` is
non-zero across all 384 channels.

### Phase 2 — Control API, JSON loader, and per-chunk slicing

Define the in-memory control representation in `demo_html.py`:

- `hint: FloatTensor [N_motion_frames_total, 55, 3]` — target world positions
  for the **entire generation** (sparse — most entries are zero).
- `mask: FloatTensor [N_motion_frames_total, 55]` — 1 where active, else 0.
- `weight: float` — global loss scale.

The JSON loader accepts a flat list of `{frame, joint, xyz}` entries (and
optional per-entry `weight`, defaulting to 1.0). Frame indices are
**motion-frame indices in the full output sequence**, not chunk-local. The
loader scatters them into the dense full-length `hint` + `mask`.

**Per-chunk slicing happens at use time**, inside the chunked loop in
`_g_test`. For each chunk `i`:

- Motion-frame range covered by chunk `i`'s output: `[i * round_l,
  i * round_l + pose_length)` (with the first `pre_frames * vqvae_squeeze_scale = 16`
  frames being the seeded/overlap region).
- Build a chunk-local hint/mask by indexing `hint[i*round_l : i*round_l + pose_length]`
  and `mask[...]` analogously.
- **Zero out the mask in the overlap region for `i > 0`** — those frames are
  inherited as seed from the previous chunk and shouldn't be re-guided.
- If the chunk's mask is all zeros (no keyframes in this chunk's range), skip
  guidance entirely for chunk `i` and run the standard un-guided DDIM.

The per-chunk `(hint, mask)` is what gets passed into the model's guidance
hook for that chunk.

### Phase 3 — Guidance hook in `GestureDiffusion` (chunk-aware)

**The chunk loop lives in `_g_test` (trainer-side), the DDIM loop lives in
`_diffusion_reverse` (model-side).** Guidance hooks must be wired so that
each chunk's `_diffusion_reverse` call only sees its own per-chunk hint/mask.
Concretely:

1. `_g_test` builds the per-chunk `(hint, mask)` (Phase 2) and passes it down
   in `cond_['y']['control']` alongside the standard chunk inputs.
2. `GestureDiffusion.forward` / `_diffusion_reverse` reads
   `cond_['y'].get('control')` and threads it into the per-step guidance
   hook. If absent, behaviour is identical to today.
3. Inside `_diffusion_reverse`, the existing outer DDIM loop runs the
   denoiser once per step (with CFG as today) to produce `model_output`. We
   then use `predicted_origin` (already at `models/Diffusion.py:245`) to
   obtain the predicted clean latent `latents_pred_x0`. This is the `μ` we
   guide.
4. If `control` is present and `i <= guidance_max_step`, replace
   `latents_pred_x0` with
   `_spatial_guidance_step(latents_pred_x0, t, guidance_fn, control, ...)`
   before passing back to the scheduler.
5. Convert the guided `x0_hat` back into the form `scheduler.step` expects
   (`epsilon`, `sample`, or `v_prediction`) using the schedule's
   alphas/sigmas, then call `scheduler.step` exactly as today.

`_spatial_guidance_step` body (mirroring `OmniControl.guide`):

```
for _ in range(n_iters_at_this_step):
    x.requires_grad_(True)
    joints = guidance_fn(x)                       # latent → SMPLX joints (chunk-local time)
    loss   = ((joints - hint) ** 2 * mask).sum() * control.weight
    grad   = autograd.grad(loss, x)[0]
    grad   = variance_weight * grad               # classifier-guidance scaling
    x      = (x - scale * grad).detach()
return x
```

**Critical**: `guidance_fn(x)` here is the **chunk-local** latent→joints
function. It takes a latent of shape `[B, 384, 1, seq_len = 32]` and returns
SMPLX joints of shape `[B, pose_length = 128, 55, 3]`. The `hint` and `mask`
are also chunk-local (`[pose_length, 55, 3]` and `[pose_length, 55]`).

Schedule defaults (OmniControl-inspired but reduced for our heavier per-iter
cost and multiplied by the number of chunks):
- `n_iters(t)` — late steps get more iters. Start with **5** for the first 70%
  of steps, **30** for the last 30%. (OmniControl uses 10/500 but their
  per-iter cost is essentially free.)
- `scale_base = 20 / num_active_keyframes_in_chunk` (mirror `calc_grad_scale`,
  computed per chunk).
- `variance_weight`: equivalent of OmniControl's `model_variance`. For DDIM
  with the schedule already exposed, use the posterior variance at `t`
  (formula in `models/Diffusion.py` via `1 - alphas_cumprod`). Start with a
  scalar multiplier if the per-timestep variance derivation is fiddly.
- `t_stopgrad`: apply guidance through all steps.

**Crucially, no denoiser call inside the inner loop.** The denoiser is
run once per outer step (per chunk).

### Phase 4 — SMPLX-analog of "zero root gradient"

OmniControl zeros the gradient on HumanML3D's root joint because HumanML3D
representation is rooted at the origin. For us, two candidates:

- The 3 root-translation channels carried inside the lower-body latent's
  trailing 3 dims (the cumsum target — see `_g_test` translation branch).
- The 3 axis-angle channels for `global_orient` (joint 0 of the SMPLX pose).

Recommended default: zero the gradient on the **translation channels** of the
guided latent (cleanest match to "let the model decide where in the world to
be, just constrain joint *configuration* targets relative to it"). Make this
toggleable via CLI for experimentation.

### Phase 5 — Wiring and CLI

In `demo_html.py`:

- New args: `--control_json`, `--guidance_iters_early`,
  `--guidance_iters_late`, `--guidance_max_step`, `--guidance_weight`,
  `--guidance_freeze_root` (default `true`).
- Build the guidance closure (with trainer-bound decoders/SMPLX/etc.) once
  before calling the model, then pass through `condition_dict['y']` alongside
  the control dict.
- When `--control_json` is unset, do not build the closure and do not change
  the call path — guarantee identical output to today.

### Phase 6 — Tuning sweep

Pick one held-out audio (default `demo/examples/2_scott_0_1_1.wav`, which has a
matching GT npz). Define a small multi-joint hint (e.g., left wrist + right
wrist at 3-5 evenly spaced frames at positions sampled from the GT). Sweep
`(n_iters_early, n_iters_late, scale_base, max_step)`. Pick the combo that
drives control loss down without visibly degrading the rest of the body. Record
findings inline at the top of `demo_html.py` for future reference.

## Differences vs OmniControl's MDM base

| Aspect | OmniControl (MDM) | GestureLSM (this plan) |
|---|---|---|
| Sampler | DDPM, predicts x0 | DDIM (HuggingFace scheduler) |
| Sequence handling | Whole motion (≤196 frames) in one DDIM pass | **Chunked rolling window**: 128-frame chunks, sequential, 16-frame seed overlap |
| Guidance scope | Whole motion at once | **Per chunk only** — keyframes in chunk `i` cannot influence chunks `< i` |
| Motion representation | HumanML3D (263-d joints+root, per frame) | RVQVAE latents (128-d × 3 body parts, ~4× time-compressed) |
| Skeleton | 22 joints (SMPL body) | 55 joints (SMPLX body+face+hands) |
| Per-iter cost in guidance loop | `recover_from_ric` only (deterministic FK, no NN) | 3 × RVQVAE decoder + 6D→axis-angle → SMPLX LBS + backward (~170 ms / 128-frame chunk on GPU) |
| `n_guide_steps` feasible | 10 early / 500 late | ~5 early / ~30 late per chunk (start; tune in Phase 6) |
| Total guidance cost for a 60s clip | 1 DDIM pass | ~16 DDIM passes (one per chunk) |
| Codebook in decode | None | RVQ exists, but **bypassed at inference for guidance** via new `decode_continuous` helper. STE not used. |
| Translation | Root in HumanML3D | `cumsum` of predicted velocity for X/Z; absolute Y |
| Root gradient zeroing | `grad[..., 0] = 0` on the root joint of the rep | Recommend zeroing translation channels of the latent; toggleable |
| Realism module | Trained ControlNet branch | None (out of scope per user) |

## Challenges & risks

1. **Per-iteration cost is the bottleneck.** Backward through 3 RVQVAE
   decoders + SMPLX LBS is non-trivial (~170 ms for a 128-frame chunk on
   GPU). Mitigation: scale `n_iters` down vs OmniControl, optionally use
   `torch.utils.checkpoint` on the decoders.
2. **Total cost multiplies by chunk count.** A 60-second clip runs ~16 chunked
   DDIM passes; guidance adds the per-iter cost on every chunk that contains
   keyframes. Skip guidance entirely for chunks with empty masks (Phase 2)
   to claw back time.
3. **Causal-only control.** A keyframe at motion frame `f` can affect chunk
   `f // 112` and later chunks (via the seed), but never earlier chunks.
   Document this clearly in the CLI help. If a keyframe at frame 0 is
   desired, it must be set in chunk 0.
2. **Latent temporal compression (×4)** — control at motion frame `t` actually
   touches latent timestep `~t/4`; the decoder receptive field spreads the
   gradient over neighbouring frames. Verify in Phase 6 that the controlled
   *motion frame* moves toward the target (not a neighbour).
3. **Cumulative translation** — controlling absolute world position depends on
   integrated velocity, so small early changes propagate forward. The
   per-channel zeroing of translation gradient (Phase 4) sidesteps this for
   joint-configuration control; for absolute world-position control of an
   end-effector, accept the dynamics and weight the loss accordingly.
4. **Stochastic / STE concerns avoided by design** — the guidance path uses
   `decode_continuous` (skips quantizer) instead of `latent2origin`. No
   Gumbel sampling, no STE bias. The un-guided demo path keeps using
   `latent2origin` as before, so no regression.
5. **`scheduler.step` API for guided x0** — DDIM may expect an `epsilon`
   prediction even though we have a guided `x0`. Conversion is straightforward
   using the alphas, but must be coded carefully against the
   `scheduler.config.prediction_type`. Read this for the chosen config before
   coding Phase 3.
6. **CFG inside the guidance forward** — wasteful (we'd double the cost of an
   already-expensive backward). Recommend running the denoiser once with CFG
   to produce `model_output → predicted_origin`, then guidance uses the
   resulting `x0_hat` without any further denoiser calls.
7. **No realism branch** — pure gradient guidance can introduce "stiff arm to
   target" artifacts. Sets user expectations; if quality is unacceptable, the
   natural next step is training a control adapter.
8. **Multi-joint loss balance** — if one joint has a much larger error it can
   dominate the gradient. Consider per-active-joint normalization
   (`/ mask.sum()`) — OmniControl uses `torch.norm(... dim=-1)` then `.sum()`,
   which already partially balances. Match their formulation as the baseline.

## Verification

1. **Phase 1 gradient unit test**: random `[B, 384, 1, T]` latent with
   `requires_grad=True`, run the full `latent_to_joints` helper, sum, call
   `.backward()`, assert `latent.grad.abs().mean() > 0` for all 384 channels.
2. **Regression test** with `--control_json` unset: produce `generation.html`
   with the same seed and audio as today's `demo_html.py`; assert SMPLX joints
   are bit-identical.
3. **Loss-decrease test**: with a small hint inside a single chunk, print
   loss at every inner iteration; it should monotonically decrease for the
   early outer steps and plateau at the end.
4. **Chunk-boundary test**: place keyframes at frames just before, on, and
   just after a chunk boundary (e.g., frames 110, 112, 114 if `round_l =
   112`). Confirm: (a) the keyframe at frame 110 is hit (it's in chunk 0's
   non-overlap region); (b) the keyframe at frame 112 is hit (it's chunk
   1's first non-seed frame); (c) the overall motion is continuous across
   the boundary (no visible jump).
5. **Skip-chunk test**: put a keyframe only in chunk 3 (e.g., frame 400).
   Confirm chunks 0–2 take the un-guided path (run a quick `print` or
   timing log) and chunk 3 alone gets guidance.
6. **Single-keyframe sanity inside the multi-joint setup**: control just the
   left wrist at one frame to a target 0.5 m from the un-guided position.
   Re-render and confirm the wrist visibly moves while the rest of the body
   stays reasonable.
5. **Multi-joint sanity**: control left wrist + right wrist + one foot at
   three different frames spanning at least two chunks; same checks.
7. **Hyperparameter sweep table** (Phase 6): tabulate `n_iters_early ×
   n_iters_late × scale_base × max_step` against final loss and a qualitative
   "does it look natural" mark.

## Open items (to confirm before coding)

- **`scheduler.config.prediction_type`** for `configs/diffuser_rvqvae_128.yaml`
  — read once, determine whether `scheduler.step` wants `epsilon`, `sample`,
  or `v_prediction`. Conversion factor between guided `x0_hat` and the
  required form falls out of the schedule alphas.
- **Default audio for regression and tuning**: `2_scott_0_1_1.wav` (matched GT
  npz already in `demo/examples/`).
- **`decode_continuous` vs `latent2origin` divergence sanity check** — during
  initial integration, render one generation through both paths and compare
  joint MSE. They should be close but not identical (because of the Gumbel
  snap in `latent2origin`). This is expected; it just confirms the decoder
  generalizes to the un-snapped continuous values.
