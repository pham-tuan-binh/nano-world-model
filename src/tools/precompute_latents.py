"""Precompute VAE latents for a LeRobot dataset into a per-episode cache.

The world model trains in a frozen SD-VAE latent space, so latents can be
computed once and reused. This removes per-step video decode (the throughput
bottleneck for AV1 LeRobot datasets) and the per-step VAE encode, making
training GPU-bound and freeing memory for larger batches.

For each episode we store two fp16 arrays of shape [T, C, h, w]:
  ep{idx:05d}_mean.npy  and  ep{idx:05d}_std.npy
holding the posterior mean/std already multiplied by the VAE scaling_factor.
At train time load_latents() draws ``mean + std * noise``, reproducing the
training-time ``posterior.sample() * scaling_factor`` exactly.

The precompute reads each episode sequentially (fast decode, no random seeks).

Usage:
  PYTHONPATH=src python src/tools/precompute_latents.py \
      --repo_id binhpham/naive-bench \
      --image_key observation.images.overhead_camera \
      --out $LATENT_CACHE/naive-bench/overhead_res256
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from diffusers import AutoencoderKL  # noqa: E402

from wm_datasets.data_source import create_data_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--image_key", default="observation.images.overhead_camera")
    ap.add_argument("--video_backend", default="pyav")
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--resize_mode", default="stretch", choices=["stretch"])
    ap.add_argument("--chunk", type=int, default=64, help="frames per VAE encode batch")
    ap.add_argument("--out", required=True, help="output cache directory")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading VAE {args.vae} ...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    vae.requires_grad_(False)
    sf = float(vae.config.scaling_factor)
    print(f"scaling_factor={sf}", flush=True)

    ds = create_data_source(
        dataset_name="lerobot",
        data_path=args.repo_id,
        root=args.root,
        image_key=args.image_key,
        video_backend=args.video_backend,
    )
    n_eps = ds.get_num_trajectories()
    print(f"{n_eps} episodes -> {args.out}", flush=True)

    size = (args.image_size, args.image_size)
    t0 = time.time()
    total_frames = 0
    for ep in range(n_eps):
        mean_path = os.path.join(args.out, f"ep{ep:05d}_mean.npy")
        std_path = os.path.join(args.out, f"ep{ep:05d}_std.npy")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            continue

        T = ds.get_seq_length(ep)
        frames = ds.load_visual_frames(index=ep, start=0, end=T, step=1)  # [T,C,H,W] in [0,1]
        if tuple(frames.shape[-2:]) != size:
            frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
        frames = frames * 2.0 - 1.0  # [-1, 1]

        means, stds = [], []
        with torch.no_grad():
            for i in range(0, T, args.chunk):
                x = frames[i:i + args.chunk].to(device)
                post = vae.encode(x).latent_dist
                means.append((post.mean * sf).to(torch.float16).cpu().numpy())
                stds.append((post.std * sf).to(torch.float16).cpu().numpy())

        mean = np.concatenate(means, axis=0)  # [T, C, h, w] fp16
        std = np.concatenate(stds, axis=0)
        # atomic-ish write
        np.save(mean_path + ".tmp.npy", mean)
        np.save(std_path + ".tmp.npy", std)
        os.replace(mean_path + ".tmp.npy", mean_path)
        os.replace(std_path + ".tmp.npy", std_path)

        total_frames += T
        if ep % 10 == 0 or ep == n_eps - 1:
            dt = time.time() - t0
            fps = total_frames / dt if dt > 0 else 0
            print(f"ep {ep+1}/{n_eps}  T={T}  latent={tuple(mean.shape[1:])}  "
                  f"{fps:.0f} fps  elapsed {dt:.0f}s", flush=True)

    print(f"Done. {total_frames} frames in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
