#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Replay static images on a monitor and record one AEDAT4 per image from DAVIS346.
# Implements the paper-style RCLS (Repeated Closed-Loop Smooth) 45° diagonal motion.
#
# Usage:
#   python replay_images_to_events.py \
#     --images-dir dataset_rgb/images \
#     --out-dir replayed_events/aedat \
#     --pattern RCLS --hz 60 --loops 6 --step 10 --speed 200 --transition 2.0
#

import os
import sys
import time
import glob
import csv
import argparse
from typing import Tuple, List

import numpy as np
from PIL import Image
import pygame
import dv_processing as dv


# ------------------------------- Camera helpers -------------------------------

def open_camera(mode: str):
    mode = (mode or "open").lower()
    if mode == "davis":
        cam = dv.io.camera.DAVIS()
    else:
        cam = dv.io.camera.open()  # first available
    print(f"[camera] Opened: {cam.getCameraName()}")
    return cam

def make_writer(path: str, cam):
    # Auto-configure to write all streams provided by the camera
    return dv.io.MonoCameraWriter(path, cam)

def batch_len(batch):
    """Return number of items in a dv_processing batch (EventStore/list/etc.)."""
    if batch is None:
        return 0
    # dv_processing.EventStore has a .size() method
    if hasattr(batch, "size") and callable(batch.size):
        return batch.size()
    # lists/tuples/ndarrays
    try:
        return len(batch)
    except Exception:
        return 0

# ------------------------------- Display helpers ------------------------------

def init_display(fullscreen: bool = True):
    pygame.display.init()
    # Init font module explicitly and use default font to avoid fc-list
    pygame.font.init()

    info = pygame.display.Info()
    size = (info.current_w, info.current_h)
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode(size, flags)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    # Use default font instead of SysFont to avoid system font scan
    try:
        font = pygame.font.Font(None, 28)
    except Exception as e:
        print(f"[warn] font init failed: {e}")
        font = None
    return screen, clock, size, font

def load_image_surface(img_path: str, fit: str):
    """
    Load an image as a pygame.Surface according to `fit` mode.
    Returns (surface, mapping) where mapping has keys describing the transform:
      - mode: 'resize' | 'letterbox' | 'none'
      - in_w, in_h: original image size
      - out_w, out_h: rendered surface size
      - scale: scalar (resize) or min-scale (letterbox), 1.0 for none
      - pad_x, pad_y: top-left padding for letterbox (else 0)
    """
    im = Image.open(img_path).convert("RGB")
    iw, ih = im.size

    if fit == "resize":
        target = 512
        im2 = im.resize((target, target), Image.BICUBIC)
        surf = pygame.image.frombuffer(im2.tobytes(), (target, target), "RGB")
        mapping = dict(mode="resize", in_w=iw, in_h=ih, out_w=target, out_h=target,
                       scale=min(target/iw, target/ih), pad_x=0, pad_y=0)

    elif fit == "letterbox":
        target = 512
        scale = min(target / iw, target / ih)
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        im2 = im.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (target, target), (0, 0, 0))
        pad_x = (target - nw) // 2
        pad_y = (target - nh) // 2
        canvas.paste(im2, (pad_x, pad_y))
        surf = pygame.image.frombuffer(canvas.tobytes(), (target, target), "RGB")
        mapping = dict(mode="letterbox", in_w=iw, in_h=ih, out_w=target, out_h=target,
                       scale=scale, pad_x=pad_x, pad_y=pad_y)

    else:  # 'none' → original size
        # No scaling; use native image resolution
        surf = pygame.image.frombuffer(im.tobytes(), (iw, ih), "RGB")
        mapping = dict(mode="none", in_w=iw, in_h=ih, out_w=iw, out_h=ih,
                       scale=1.0, pad_x=0, pad_y=0)

    return surf, mapping

# ------------------------------- Motion generators ----------------------------

def tri_wave(t, T):
    # 0..1..0 over period T
    phase = (t % T) / T
    return 2*phase if phase <= 0.5 else 2*(1 - phase)

def generate_sweep_positions(img_size: Tuple[int,int],
                             disp_size: Tuple[int,int],
                             duration_s: float,
                             hz: int,
                             pattern: str = "HV") -> List[Tuple[int,int]]:
    """Simple horizontal/vertical sweeps (fallback/baseline)."""
    iw, ih = img_size; dw, dh = disp_size
    frames = max(1, int(round(duration_s * hz)))
    seq: List[Tuple[int,int]] = []

    max_x = max(0, iw - dw)
    max_y = max(0, ih - dh)
    cx = (dw - iw) // 2
    cy = (dh - ih) // 2

    if "H" in pattern:
        N = frames // (2 if "V" in pattern else 1)
        for t in range(N):
            a = tri_wave(t, N)
            x = int(a * max_x) if max_x > 0 else 0
            seq.append((cx - x, cy))
    if "V" in pattern:
        N = frames - len(seq)
        for t in range(N):
            a = tri_wave(t, N)
            y = int(a * max_y) if max_y > 0 else 0
            seq.append((cx, cy - y))
    return seq

def rcls_positions(img_size: Tuple[int,int],
                   disp_size: Tuple[int,int],
                   loops: int = 6,
                   step_px: int = 10,
                   speed_px_per_s: int = 200,
                   hz: int = 60) -> List[Tuple[int,int]]:
    """
    RCLS: four 45° diagonal segments per loop, repeated `loops` times.
    Waypoints (relative): (0,0)->(s,-s)->(2s,0)->(s,s)->(0,0)
    With speed ≈ 200 px/s and s=10 px, each segment ~50 ms; 4 seg * 6 loops ≈ 1.2 s total motion.
    """
    iw, ih = img_size; dw, dh = disp_size
    # Center the 512×512 on screen; motion is relative to center
    cx = (dw - iw) // 2
    cy = (dh - ih) // 2

    rel_pts = [(0, 0), ( step_px, -step_px),
               (2*step_px, 0),
               ( step_px,  step_px),
               (0, 0)]
    seq: List[Tuple[int,int]] = []
    dt = 1.0 / float(hz)
    vx = float(speed_px_per_s)
    vy = float(speed_px_per_s)

    for _ in range(loops):
        for (x0, y0), (x1, y1) in zip(rel_pts[:-1], rel_pts[1:]):
            dx, dy = (x1 - x0), (y1 - y0)
            # diagonal; time to travel based on per-axis speed
            seg_t = max(abs(dx) / vx if vx > 0 else 0.05,
                        abs(dy) / vy if vy > 0 else 0.05)
            steps = max(1, int(round(seg_t / dt)))
            for s in range(1, steps + 1):
                x = x0 + (s / steps) * dx
                y = y0 + (s / steps) * dy
                seq.append((cx + int(round(x)), cy + int(round(y))))
    return seq


# ------------------------------- Replay core ----------------------------------

def record_one_image(cam,
                     screen, clock, font, disp_size,
                     img_path: str,
                     aedat_path: str,
                     pattern: str,
                     hz: int,
                     settle_s: float,
                     loops: int,
                     step: int,
                     speed: int,
                     transition_s: float,
                     simple_duration_s: float,
                     fit: str="resize"):
    """Display one image with motion, record to AEDAT, return (events, frames) counts."""
    # Load image
    surf, mapping = load_image_surface(img_path, fit=fit)
    iw, ih = surf.get_size()


    # Motion sequence
    if pattern == "RCLS":
        positions = rcls_positions((iw, ih), disp_size, loops=loops, step_px=step,
                                   speed_px_per_s=speed, hz=hz)
    else:
        positions = generate_sweep_positions((iw, ih), disp_size,
                                             duration_s=simple_duration_s,
                                             hz=hz, pattern=pattern)

    # Settle (static)
    screen.fill((0, 0, 0))
    cx = (disp_size[0] - iw) // 2
    cy = (disp_size[1] - ih) // 2
    screen.blit(surf, (cx, cy))
    label = font.render(f"{os.path.basename(img_path)}  settle...", True, (220, 220, 220))
    screen.blit(label, (20, 20))
    pygame.display.flip()
    time.sleep(max(0.0, settle_s))

    writer = make_writer(aedat_path, cam)

    events_written = 0
    frames_written = 0
    imu_written = 0
    trg_written = 0

    try:
        for (ox, oy) in positions:
            # ESC quits
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            # draw
            screen.fill((0, 0, 0))
            screen.blit(surf, (ox, oy))
            pygame.display.flip()

            # events
            ev = cam.getNextEventBatch()
            if batch_len(ev) > 0:
                writer.writeEvents(ev)
                events_written += batch_len(ev)

            # frames
            fr = cam.getNextFrame()
            if fr is not None:
                writer.writeFrame(fr)
                frames_written += 1

            # IMU
            imu = cam.getNextImuBatch()
            if batch_len(imu) > 0:
                for m in imu:
                    writer.writeImu(m)
                    imu_written += 1

            # triggers
            trg = cam.getNextTriggerBatch()
            if batch_len(trg) > 0:
                for tr in trg:
                    writer.writeTrigger(tr)
                    trg_written += 1

            clock.tick(hz)

    finally:
        # Close file cleanly
        del writer

    # Transition pause between images (paper: ~2 s)
    if pattern == "RCLS":
        pygame.time.wait(int(transition_s * 1000))

    return events_written, frames_written, imu_written, trg_written


def main():
    ap = argparse.ArgumentParser(description="Replay RGB images on LCD and record DAVIS AEDAT per image (RCLS-compatible).")
    ap.add_argument("--images-dir", required=True, help="Folder with RGB images")
    ap.add_argument("--out-dir", default="replayed_events/aedat", help="Output folder for .aedat4 files")
    ap.add_argument("--camera", default="open", choices=["open", "davis"], help="Which camera open() strategy to use")
    ap.add_argument("--hz", type=int, default=60, help="Display refresh/animation rate")
    ap.add_argument("--pattern", default="RCLS", choices=["RCLS", "H", "V", "HV"], help="Motion pattern")
    # Paper params (RCLS)
    ap.add_argument("--loops", type=int, default=6, help="RCLS loops (paper: 6)")
    ap.add_argument("--step", type=int, default=10, help="RCLS step in pixels between waypoints (paper: 10)")
    ap.add_argument("--speed", type=int, default=200, help="RCLS per-axis speed in px/s (paper: ~200)")
    ap.add_argument("--transition", type=float, default=2.0, help="Pause (s) between images (paper: ~2.0)")
    ap.add_argument("--settle", type=float, default=0.4, help="Static settle time before motion (s)")
    # Simple sweep fallback
    ap.add_argument("--duration", type=float, default=2.5, help="Duration for simple H/V/HV motion (s)")
    ap.add_argument("--meta-csv", default=None, help="Optional meta CSV path (default: <out-dir>/../meta/meta.csv)")
    ap.add_argument("--fit", choices=["resize", "letterbox", "none"], default="resize",
    help="How to place images: 'resize' (stretch to 512x512), 'letterbox' (preserve aspect into 512x512), or 'none' (use original size)")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_dir = os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "meta")
    os.makedirs(meta_dir, exist_ok=True)
    meta_csv = args.meta_csv or os.path.join(meta_dir, "meta.csv")

    images = sorted(glob.glob(os.path.join(args.images_dir, "*.*")))
    if not images:
        print(f"[error] No images found in {args.images_dir}")
        sys.exit(1)

    cam = open_camera(args.camera)
    screen, clock, disp_size, font = init_display(fullscreen=True)

    # write header
    if not os.path.isfile(meta_csv):
        with open(meta_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["image", "aedat", "pattern", "hz", "loops", "step", "speed_px_s", "transition_s",
                 "settle_s", "events", "frames", "imu", "triggers", "wall_start_us", "wall_end_us"]
            )

    try:
        for i, ipath in enumerate(images, 1):
            base = os.path.splitext(os.path.basename(ipath))[0]
            aedat_path = os.path.join(args.out_dir, f"{base}.aedat4")

            # status line
            screen.fill((0, 0, 0))
            label = font.render(f"[{i}/{len(images)}] {base}", True, (200, 200, 200))
            screen.blit(label, (20, 20))
            pygame.display.flip()

            wall_start = int(time.time() * 1e6)

            e_cnt, f_cnt, imu_cnt, trg_cnt = record_one_image(
                cam, screen, clock, font, disp_size,
                img_path=ipath, aedat_path=aedat_path,
                pattern=args.pattern, hz=args.hz, settle_s=args.settle,
                loops=args.loops, step=args.step, speed=args.speed,
                transition_s=args.transition, simple_duration_s=args.duration,fit=args.fit
            )

            wall_end = int(time.time() * 1e6)

            with open(meta_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    ipath, aedat_path, args.pattern, args.hz, args.loops, args.step, args.speed,
                    args.transition, args.settle, e_cnt, f_cnt, imu_cnt, trg_cnt, wall_start, wall_end
                ])

            # brief on-screen confirmation
            screen.fill((0, 0, 0))
            msg = font.render(f"Saved {os.path.basename(aedat_path)}  ev={e_cnt:,} fr={f_cnt}", True, (150, 220, 150))
            screen.blit(msg, (20, 20))
            pygame.display.flip()
            pygame.time.wait(250)

    except KeyboardInterrupt:
        print("\n[info] Interrupted. Closing.")

    finally:
        pygame.display.quit()

    print(f"[done] AEDAT files: {args.out_dir}")
    print(f"[done] Meta CSV:   {meta_csv}")


if __name__ == "__main__":
    main()
