"""Trim one .npz + .wav pair to N motion frames.

Time-axis arrays in the npz (poses, trans, expressions) are sliced to N rows.
The audio is sliced to N/fps seconds at its original sample rate. Static arrays
(betas, model, gender, mocap_frame_rate) are copied unchanged.

Note on the choice of N: the dataloader does integer-second truncation
(`pose_frames // pose_fps`). To produce exactly one inference chunk (128 motion
frames = 32 latent tokens), pass `--frames 150` — that's the smallest multiple
of 30 that survives the dataloader's clamp and still yields one chunk.

Usage:
    python visualize/trim.py                                    # defaults: 150 frames, 2_scott_0_1_1
    python visualize/trim.py --frames 300
    python visualize/trim.py --basename 2_scott_0_5_5 --frames 150
"""
import argparse
import os

import numpy as np
import soundfile as sf


def trim_npz(src, dst, n_frames):
    d = np.load(src, allow_pickle=True)
    # Anchor on the largest known time-axis length; anything matching it gets sliced.
    time_len = max(
        d[k].shape[0] for k in d.files
        if hasattr(d[k], "shape") and len(d[k].shape) >= 1
    )
    out = {}
    for k in d.files:
        v = d[k]
        if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == time_len:
            out[k] = v[:n_frames]            # time-axis array → slice
        else:
            out[k] = v                       # static metadata → copy
    np.savez(dst, **out)
    return {k: out[k].shape if hasattr(out[k], "shape") else type(out[k]).__name__
            for k in out}


def trim_wav(src, dst, n_frames, fps):
    wav, sr = sf.read(src)                   # original sample rate preserved
    n_samples = n_frames * sr // fps          # integer; e.g. 150 * 16000 // 30 = 80000
    n_samples = min(n_samples, len(wav))
    sf.write(dst, wav[:n_samples], sr)
    return sr, n_samples, n_samples / sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="visualize/input")
    p.add_argument("--basename", default="2_scott_0_1_1",
                   help="Common basename for .npz and .wav (without extension).")
    p.add_argument("--frames", type=int, default=150,
                   help="Number of motion frames to keep. Default 150 = smallest value "
                        "that gives exactly one inference chunk through the model.")
    p.add_argument("--fps", type=int, default=30,
                   help="Motion frame rate; must match the model's pose_fps.")
    p.add_argument("--suffix", default="_128f",
                   help="Appended to basename for the output files.")
    args = p.parse_args()

    npz_src = os.path.join(args.input_dir, args.basename + ".npz")
    wav_src = os.path.join(args.input_dir, args.basename + ".wav")
    npz_dst = os.path.join(args.input_dir, args.basename + args.suffix + ".npz")
    wav_dst = os.path.join(args.input_dir, args.basename + args.suffix + ".wav")

    print(f"Trimming to {args.frames} frames ({args.frames / args.fps:.4f}s @ {args.fps} fps)")
    print(f"  npz: {npz_src}  →  {npz_dst}")
    shapes = trim_npz(npz_src, npz_dst, args.frames)
    for k, s in shapes.items():
        print(f"    {k:18s} {s}")

    print(f"  wav: {wav_src}  →  {wav_dst}")
    sr, n, dur = trim_wav(wav_src, wav_dst, args.frames, args.fps)
    print(f"    sr={sr} samples={n} duration={dur:.4f}s")


if __name__ == "__main__":
    main()
