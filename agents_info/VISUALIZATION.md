# Visualization Pipeline

End-to-end docs for the audio-to-gesture demo and 3D viewer in this fork.
Covers `gen.py`, `visualize/output/visualize.html`, `visualize/trim.py`, the
dataloader hook, and the file layout / caching behaviour.

## TL;DR

```bash
# 1) generate (writes gen_*.npz + speech.wav into visualize/output/)
python gen.py

# 2) open the static viewer in Chrome/Firefox
xdg-open visualize/output/visualize.html
# drag visualize/output/  (the folder) onto the drop zone
```

First run cold (downloads Whisper, runs MFA, builds caches) ≈ **140 s**.
Subsequent runs on the same audio (caches hit) ≈ **24 s** (~6× faster).
Generated motion length ≈ 60 s for the default audio (1808 frames @ 30 fps).

## File layout

```
GestureLSM/
├── gen.py                         # CLI: audio + GT → joints + npz/wav
├── dataloaders/beat_sep_single.py # patched: reads args.pose_file_path
└── visualize/
    ├── trim.py                    # minimal-length test helper (150 frames)
    ├── input/
    │   ├── 2_scott_0_1_1.wav      # default audio
    │   ├── 2_scott_0_1_1.npz      # default GT (poses + trans + betas + …)
    │   └── 2_scott_0_1_1_128f.{wav,npz}    # trimmed (one-chunk test)
    └── output/
        ├── visualize.html         # static viewer (hand-written, ~440 lines)
        ├── gen_rotation.npz       # canonical generation
        ├── gen_position.npz       # viewer-only (joints from SMPLX FK)
        ├── gt_position.npz        # viewer-only (joints from SMPLX FK)
        └── speech.wav             # copy of input audio
```

The viewer file (`visualize/output/visualize.html`) lives **only** in
`visualize/output/` — `gen.py` does NOT regenerate it. Edit it in place.

## gen.py

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--audio` | `visualize/input/2_scott_0_1_1.wav` | Input speech audio |
| `--gt` | `<audio>.npz` (sibling) | GT motion `.npz` with `poses`, `trans`, `betas` |
| `--out` | `visualize/output` | Output folder |
| `--config / -c` | `configs/shortcut_rvqvae_128_hf.yaml` | Model config |
| `--no-cache` | — | Force-rebuild MFA TextGrid + dataloader LMDB cache |

### Pipeline (inside `HTMLTrainer`)

1. **MD5-hash input audio** → stable per-audio cache dir
   `outputs/audio2pose/custom/audio_<10>_<6>/`.
2. **ASR** with Whisper-tiny.en → text (skipped if TextGrid exists).
3. **MFA align** → TextGrid (skipped if file exists). Slowest cold step (~25 s).
4. **`CustomDataset(args, "test")`** reads:
   - audio from `args.audio_file_path`
   - TextGrid from `args.textgrid_file_path`
   - **pose from `args.pose_file_path`** ← see "Dataloader hook" below
   and writes an LMDB cache (skipped if cache dir exists).
5. **Run inference** → `rec_pose_aa [N,165]`, `rec_trans [N,3]`.
6. **SMPLX FK** with `betas` from the GT → joints `[N, 55, 3]`.
7. **Write outputs**: `gen_rotation.npz`, `gen_position.npz`, `gt_position.npz`,
   `speech.wav`.

### Output file schemas

`gen_rotation.npz` (canonical model output, lossless):
| Key | Shape | Dtype |
|---|---|---|
| `poses` | `[N, 165]` | float32 (SMPLX axis-angle, 55 joints × 3) |
| `trans` | `[N, 3]` | float64 (world translation) |
| `betas` | `[300]` | float32 (body shape — from GT, not generated) |
| `fps` | `()` | int32 |

`gen_position.npz` / `gt_position.npz` (viewer-only, smaller):
| Key | Shape | Dtype |
|---|---|---|
| `joints` | `[N, 55, 3]` | float32 (world coords, SMPLX 55-joint tree) |
| `fps` | `()` | int32 |

`speech.wav`: byte-for-byte copy of the input audio.

## visualize.html

Self-contained Plotly viewer. ~440 lines. No build step. Loaded from `file://`.

### What it does

- On first paint: shows a **drop zone**.
- User **drags the output folder** (or the three files) onto it.
- Browser reads `gen_position.npz`, `gt_position.npz`, `speech.wav` via the File API.
- An inline **NPZ + NPY parser** (handles ZIP_STORED entries and the standard
  NumPy v1/v2/v3 header formats; skips unsupported dtypes like `<U9`) extracts
  the `joints` arrays.
- Renders a Plotly 3D scene:
  - **Generation**: blue bones, red joints.
  - **Ground truth**: green bones, orange joints. Hide-able via "Hide GT" button.
- Both skeletons re-anchored to their frame-0 pelvis (so they start at origin
  and diverge from there).
- Audio drives the animation via `requestAnimationFrame` polling
  `<audio>.currentTime`. Spacebar toggles play/pause. Slider scrubs.

### Browser limitations to know

- Must open in **Chrome or Firefox**. VSCode's Simple Browser blocks `file://`
  drag-and-drop and file pickers.
- The drop zone requires the **folder picker / drag-drop**; you cannot use
  `fetch('./gen_position.npz')` on `file://` — browsers block binary fetches
  from local files.

### Why position-only is enough for the viewer

The viewer never needs to run SMPLX. `gen_rotation.npz` is the canonical
generator output (you'd use it for re-rendering with different body shapes, for
Blender export, for OmniControl integration, etc.). `gen_position.npz` is the
*precomputed forward-kinematics* result — it's all the viewer needs.

## visualize/trim.py

Helper that creates minimum-length test inputs. Lets you run one chunk
(32 latent tokens) through the model.

### Why 150 frames, not 128

You'd expect "128 motion frames = 32 latent tokens" to be the minimum. But the
dataloader does integer-second truncation
(`dataloaders/beat_sep_single.py:490`):

```python
round_seconds_skeleton = pose_frames // pose_fps         # 128 // 30 = 4
cut_length = round_seconds_skeleton * pose_fps           # 4 * 30 = 120 ← short!
```

So 128 input frames → dataloader clamps to 120 → `roundt = (120-16)//112 = 0` →
empty chunk list → crash.

The smallest multiple of 30 ≥ 128 is **150 (= 5 s)**:
- 150 → after `%8` trim in `_g_test` → **144** → `(144-16)//112 = 1` ✓
- Model processes exactly **128 motion frames** (32 latent tokens). The extra
  22 are buffer the model never sees.

### Usage

```bash
# default: trims 2_scott_0_1_1 to 150 frames; outputs _128f.{npz,wav}
python visualize/trim.py

# arbitrary length
python visualize/trim.py --frames 300

# different sample
python visualize/trim.py --basename 2_scott_0_5_5 --frames 150
```

Then:
```bash
python gen.py --audio visualize/input/2_scott_0_1_1_128f.wav \
              --out visualize/output_128f
```

GT auto-resolves to `..._128f.npz`. Output dir gets the three npz/wav files;
copy `visualize.html` into it once if you want a self-contained folder.

## Dataloader hook: `args.pose_file_path`

The demo dataloader `dataloaders/beat_sep_single.py` historically hardcoded
its pose source on line 33:
```python
self.default_pose_file = "./demo/examples/2_scott_0_1_1.npz"
```

We threaded a single-line override:
```python
self.default_pose_file = getattr(args, "pose_file_path", None) or "./demo/examples/2_scott_0_1_1.npz"
```

`gen.py` sets `args.pose_file_path` to the GT path you pass (or auto-derive)
**before** constructing the dataset. Without this, trimmed `.npz` files in
`visualize/input/` would be silently ignored and the dataloader would always
read the un-trimmed example.

This affects only `beat_sep_single.py`. Other dataloaders
(`beat_sep_lower.py`, `mix_sep.py`, etc.) used by training and the DDIM config
are unaffected.

## Caching

Three levels of cache, all on-disk, all stable per-audio:

| Cache | Path | Skip condition | Cold time saved |
|---|---|---|---|
| HF Hub offline mode | `~/.cache/huggingface/hub/models--openai--whisper-tiny.en/` | auto-detected at `gen.py` startup; sets `HF_HUB_OFFLINE=1` | ~5–10 s |
| MFA TextGrid | `outputs/audio2pose/custom/audio_<hash>/tmp.TextGrid` | file exists | ~20–25 s |
| Dataloader LMDB | `outputs/audio2pose/custom/audio_<hash>/test/smplxflame_30_cache/` | dir exists (dataloader's own check) | ~3–5 s |

The per-audio dir name is `audio_<md5_of_audio:10>_<md5_of_pose_path:6>`. Any
byte change to the audio file or any change to the GT path invalidates the
cache automatically.

Force-rebuild everything with `python gen.py --no-cache ...`.

### Cold vs warm timing (1-chunk run)

| Phase | Cold | Warm |
|---|---|---|
| Python + CUDA init | ~12 s | ~12 s |
| Whisper HF validation | ~5 s | 0 (offline) |
| MFA align | ~25 s | 0 (cached) |
| LMDB cache build | ~3 s | 0 (cached) |
| Checkpoint loading | ~20 s | ~20 s |
| Inference (1 chunk) | ~1 s | ~1 s |
| **Total** | **~140 s** | **~24 s** |

The remaining ~20 s of warm-run cost is **model loading**. The only way to
amortize that further is to keep a long-lived Python process (Jupyter/REPL/
server) — not currently implemented.

## Common gotchas

- **`with-proxy python gen.py` is only needed for the first run** (to download
  Whisper through fwdproxy). After that, `HF_HUB_OFFLINE=1` is auto-set and
  `python gen.py` works offline.
- **VSCode Simple Browser** silently breaks the drop zone. Always open
  `visualize.html` in Chrome/Firefox.
- **Don't trim audio shorter than 5 s** unless you also change the model's
  `pose_length` config. Below 150 motion frames you'll get
  `torch.cat: expected a non-empty list of Tensors`.
- **Default GT auto-resolution**: `--audio foo.wav` → looks for `foo.npz` in
  the same dir. Override with `--gt path/to/other.npz`.
- **Only `_hf` configs work with `gen.py`**: `shortcut_hf.yaml` and
  `shortcut_rvqvae_128_hf.yaml` use the `beat_sep_single` dataloader.
  `diffuser_rvqvae_128.yaml` and the training configs use `beat_sep_lower`
  which has a different file-loading contract.

## Future work hooks

- **`omnicontrol_spatial_guidance.md`** — separate plan for adding
  inference-time spatial guidance (OmniControl-style) on the DDIM checkpoint.
- **Model-loading amortization** — a `gen_server.py` that loads checkpoints
  once and serves multiple inference calls would cut warm-run cost from 24 s
  to ~2 s. Not yet implemented.
