#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAVIS346 AEDAT4 extractor (dv-processing 2.0.2)
- Frames  -> frames/frame_000000.png, ...
- Events  -> events/events.npy (structured array with fields: x,y,timestamp,polarity)
- Windows -> meta/frame_windows_us.csv (per-frame [t_start, t_end) for label→event alignment)
- Optional voxel grids per frame window: voxels/voxel_000000.npy with shape (B,H,W)

Usage:
  python event_process.py <file.aedat4> [--make-voxels] [--voxel-bins 5]
"""

import os, sys, csv, argparse
import numpy as np
from PIL import Image
import dv_processing as dv

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def write_frames_and_get_timestamps(aedat_path, frames_dir):
    reader = dv.io.MonoCameraRecording(aedat_path)
    ts = []
    idx = 0
    while reader.isRunning():
        f = reader.getNextFrame()
        if f is not None:
            Image.fromarray(f.image).save(os.path.join(frames_dir, f"frame_{idx:06d}.png"))
            ts.append(int(f.timestamp))  # microseconds
            idx += 1
        else:
            # advance file if no frame; drain events quickly
            _ = reader.getNextEventBatch()
            if _ is None:
                break
    return ts

def dump_all_events(aedat_path, events_npy_path):
    reader = dv.io.MonoCameraRecording(aedat_path)
    chunks = []
    n = 0
    while reader.isRunning():
        ev = reader.getNextEventBatch()
        if ev is not None:
            arr = ev.numpy()  # structured array with fields ('x','y','timestamp','polarity')
            chunks.append(arr)
            n += arr.shape[0]
        else:
            _ = reader.getNextFrame()
            if _ is None:
                break
    if not chunks:
        raise RuntimeError("No events found in the recording.")
    events = np.concatenate(chunks, axis=0)
    np.save(events_npy_path, events)
    return events

def write_frame_windows_csv(frame_ts, events, meta_csv_path):
    # define per-frame window as [prev_ts, curr_ts)
    if events is not None and events.size > 0:
        t_min = int(events['timestamp'].min())
    elif frame_ts:
        t_min = int(frame_ts[0])
    else:
        t_min = 0
    rows = []
    prev = t_min
    for i, ts in enumerate(frame_ts):
        rows.append((i, prev, int(ts), int(ts) - int(prev)))
        prev = int(ts)
    with open(meta_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "t_start_us", "t_end_us", "duration_us"])
        w.writerows(rows)
    return rows

def make_voxel_grid(events_slice, height, width, bins):
    """
    Simple voxelization: time-linear binning over [t_min, t_max).
    Output shape: (bins, height, width), dtype=float32 in [0,1] (per-pixel normalized).
    """
    if events_slice.size == 0:
        return np.zeros((bins, height, width), dtype=np.float32)
    t0 = events_slice['timestamp'].min()
    t1 = events_slice['timestamp'].max()
    if t1 == t0:
        t1 = t0 + 1  # avoid divide-by-zero
    # normalize times to [0, bins)
    t_norm = (events_slice['timestamp'] - t0) * (bins / float(t1 - t0))
    t_idx = np.clip(t_norm.astype(np.int32), 0, bins - 1)

    # polarity: map {False, True} -> {-1, +1}
    pol = np.where(events_slice['polarity'], 1.0, -1.0).astype(np.float32)
    x = events_slice['x'].astype(np.int32)
    y = events_slice['y'].astype(np.int32)

    vox = np.zeros((bins, height, width), dtype=np.float32)
    # Accumulate; for speed, do a flat index add
    for b, yy, xx, pp in zip(t_idx, y, x, pol):
        if 0 <= yy < height and 0 <= xx < width:
            vox[b, yy, xx] += pp

    # per-bin per-pixel tanh-like squash via normalization
    max_abs = np.max(np.abs(vox))
    if max_abs > 0:
        vox /= max_abs
    return vox

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aedat", type=str, help="Path to .aedat4 file")
    ap.add_argument("--make-voxels", action="store_true", help="Export per-frame voxel grids")
    ap.add_argument("--voxel-bins", type=int, default=5, help="Temporal bins for voxel grid")
    ap.add_argument("--height", type=int, default=260, help="Sensor height (DAVIS346 default=260)")
    ap.add_argument("--width", type=int, default=346, help="Sensor width  (DAVIS346 default=346)")
    args = ap.parse_args()

    in_path = args.aedat
    if not os.path.isfile(in_path):
        print(f"ERROR: file not found: {in_path}")
        sys.exit(1)

    frames_dir = "frames"
    events_dir = "events"
    meta_dir   = "meta"
    vox_dir    = "voxels"

    ensure_dirs(frames_dir, events_dir, meta_dir)
    if args.make_voxels:
        ensure_dirs(vox_dir)

    print("dv-processing version:", dv.__version__)

    # 1) Frames and timestamps
    frame_ts = write_frames_and_get_timestamps(in_path, frames_dir)
    print(f"Saved {len(frame_ts)} frames to {frames_dir}")

    # 2) Events (structured array)
    events_npy = os.path.join(events_dir, "events.npy")
    events = dump_all_events(in_path, events_npy)
    print(f"Saved {events.shape[0]} events to {events_npy}")
    # Sanity: expected fields
    # print("Event dtype:", events.dtype, "fields:", events.dtype.names)

    # 3) Frame windows CSV
    meta_csv = os.path.join(meta_dir, "frame_windows_us.csv")
    windows = write_frame_windows_csv(frame_ts, events, meta_csv)
    print(f"Wrote per-frame windows to {meta_csv}")

    # 4) Optional: per-frame voxel grids
    if args.make_voxels:
        # pre-index timestamps once for fast slicing
        t_all = events['timestamp']
        for i, (idx, t0, t1, _) in enumerate(windows):
            mask = (t_all >= t0) & (t_all < t1)
            ev_i = events[mask]
            vox = make_voxel_grid(ev_i, args.height, args.width, args.voxel_bins)
            np.save(os.path.join(vox_dir, f"voxel_{idx:06d}.npy"), vox)
        print(f"Saved {len(windows)} voxel grids to {vox_dir} with {args.voxel_bins} bins")

    print("Done.")

if __name__ == "__main__":
    main()
