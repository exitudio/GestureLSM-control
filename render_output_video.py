#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pyrender
import smplx
import torch
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Render a generated-only GestureLSM mesh video from an output folder.")
    parser.add_argument("output_dir", help="Folder with gen_rotation.npz and optional control_position.npz/speech.wav")
    parser.add_argument("--out", default=None, help="Output mp4 path. Default: <output_dir>/generated_mesh.mp4")
    parser.add_argument("--model-folder", default="datasets/hub/smplx_models")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth source frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug cap")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--control-fade", type=float, default=1.0, help="Seconds of fade before each control frame")
    parser.add_argument("--xmag", type=float, default=None, help="Orthographic horizontal magnification. Defaults to ymag * width / height.")
    parser.add_argument("--ymag", type=float, default=1.0)
    return parser.parse_args()


def load_controls(output_dir):
    path = output_dir / "control_position.npz"
    if not path.exists():
        return []
    data = np.load(path)
    frames = data["frames"].astype(np.int64)
    points = data["points"].astype(np.float32)
    joints = data["joints"].astype(np.int64) if "joints" in data else np.full(len(frames), -1)
    return [{"frame": int(f), "joint": int(j), "point": p} for f, j, p in zip(frames, joints, points)]


def active_controls(controls, frame, fps, fade_seconds):
    fade = max(1, int(round(fps * fade_seconds)))
    pts, alphas = [], []
    for c in controls:
        control_frame = c["frame"]
        start = max(0, control_frame - fade)
        if frame < start or frame > control_frame:
            continue
        progress = (frame - start) / max(1, control_frame - start)
        pts.append(c["point"])
        alphas.append(0.15 + 0.85 * progress)
    return pts, alphas


def camera_pose():
    angle = np.deg2rad(-2.0)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle), 1.0],
        [0.0, np.sin(angle), np.cos(angle), 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def light_pose():
    angle = np.deg2rad(-30.0)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle), 0.0],
        [0.0, np.sin(angle), np.cos(angle), 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def project_world_to_pixel(point, width, height, xmag, ymag):
    cam = np.linalg.inv(camera_pose()) @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    x_ndc = cam[0] / xmag
    y_ndc = cam[1] / ymag
    px = int(round((x_ndc + 1.0) * 0.5 * width))
    py = int(round((1.0 - (y_ndc + 1.0) * 0.5) * height))
    return px, py


def overlay_controls(frame_rgb, controls, source_frame, fps, fade_seconds, xmag, ymag):
    pts, alphas = active_controls(controls, source_frame, fps, fade_seconds)
    if not pts:
        return frame_rgb
    img = frame_rgb.copy()
    h, w = img.shape[:2]
    for point, alpha in zip(pts, alphas):
        x, y = project_world_to_pixel(point, w, h, xmag, ymag)
        if x < -24 or x > w + 24 or y < -24 or y > h + 24:
            continue
        overlay = img.copy()
        radius = int(round(4 + 7 * alpha))
        diamond = np.array([[x, y - radius], [x + radius, y], [x, y + radius], [x - radius, y]], dtype=np.int32)
        cv2.fillConvexPoly(overlay, diamond, (255, 0, 255))
        cv2.polylines(overlay, [diamond], True, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0)
    return img


def add_audio(video_path, audio_path, output_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-shortest",
        str(output_path),
    ], check=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rotation_path = output_dir / "gen_rotation.npz"
    if not rotation_path.exists():
        raise FileNotFoundError(rotation_path)

    final_video = Path(args.out) if args.out else output_dir / "generated_mesh.mp4"
    silent_video = final_video.with_name(final_video.stem + ".silent.mp4")

    data = np.load(rotation_path, allow_pickle=True)
    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    betas = data["betas"].astype(np.float32)
    source_fps = int(np.asarray(data["fps"])) if "fps" in data else 30
    fps = int(args.fps or source_fps)
    if args.max_frames is not None:
        poses = poses[:args.max_frames]
        trans = trans[:args.max_frames]

    model_folder = Path(args.model_folder)
    faces = np.load(model_folder / "smplx" / "SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"].astype(np.int32)
    model = smplx.create(
        str(model_folder), model_type="smplx", gender="NEUTRAL_2020",
        use_face_contour=False, num_betas=300, num_expression_coeffs=100,
        ext="npz", use_pca=False,
    ).eval()

    controls = [] if args.no_controls else load_controls(output_dir)
    print(f"Loaded {len(controls)} control points")
    frame_indices = list(range(0, len(poses), max(1, args.stride)))

    xmag = args.xmag if args.xmag is not None else args.ymag * args.width / args.height
    ymag = args.ymag
    print(f"Camera magnification: xmag={xmag:.4f}, ymag={ymag:.4f}")

    renderer = pyrender.OffscreenRenderer(args.width, args.height)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    writer = imageio.get_writer(silent_video, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    beta_t = torch.from_numpy(betas).float().unsqueeze(0)

    try:
        with torch.no_grad():
            for out_i, src_i in enumerate(frame_indices):
                pose = torch.from_numpy(poses[src_i:src_i + 1]).float()
                transl = torch.from_numpy(trans[src_i:src_i + 1]).float()
                zeros_expr = torch.zeros(1, 100, dtype=torch.float32)
                zeros_aa = torch.zeros(1, 3, dtype=torch.float32)
                smplx_out = model(
                    betas=beta_t,
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:21 * 3 + 3],
                    jaw_pose=zeros_aa,
                    leye_pose=zeros_aa,
                    reye_pose=zeros_aa,
                    left_hand_pose=pose[:, 25 * 3:40 * 3],
                    right_hand_pose=pose[:, 40 * 3:55 * 3],
                    transl=transl,
                    expression=zeros_expr,
                    return_verts=True,
                )
                vertices = smplx_out.vertices[0].cpu().numpy()
                mesh = pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(vertices=vertices, faces=faces, process=False, vertex_colors=[220, 220, 220, 255]),
                    smooth=True,
                )
                scene = pyrender.Scene(bg_color=[248, 248, 248, 255], ambient_light=[0.25, 0.25, 0.25])
                scene.add(mesh)
                scene.add(camera, pose=camera_pose())
                scene.add(light, pose=light_pose())
                color, _ = renderer.render(scene)
                color = overlay_controls(color, controls, src_i, source_fps, args.control_fade, xmag, ymag)
                writer.append_data(color)
                if out_i % 30 == 0:
                    print(f"rendered {out_i + 1}/{len(frame_indices)} frames", flush=True)
    finally:
        writer.close()
        renderer.delete()

    audio_path = output_dir / "speech.wav"
    if not args.no_audio and audio_path.exists():
        try:
            add_audio(silent_video, audio_path, final_video)
            silent_video.unlink(missing_ok=True)
        except subprocess.CalledProcessError as exc:
            print(f"Audio mux failed ({exc}); keeping silent video instead")
            if final_video.exists():
                final_video.unlink()
            silent_video.rename(final_video)
    else:
        if final_video.exists():
            final_video.unlink()
        silent_video.rename(final_video)
    print(f"Wrote {final_video}")


if __name__ == "__main__":
    main()
