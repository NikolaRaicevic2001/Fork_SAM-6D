#!/usr/bin/env python3
import argparse
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="outputs_simple")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--intrinsics_for", choices=["color", "depth"], default="color")
    ap.add_argument("--rgb_name", type=str, default="rgb.png")
    ap.add_argument("--depth_name", type=str, default="depth.png")
    ap.add_argument("--cam_name", type=str, default="camera.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.w, args.h, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    try:
        # TRUE RealSense scale: meters per raw depth unit
        depth_scale_rs = profile.get_device().first_depth_sensor().get_depth_scale()

        # Warm up
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture frames")

        # Choose intrinsics
        vsp = rs.video_stream_profile(
            color_frame.profile if args.intrinsics_for == "color"
            else depth_frame.profile
        )
        intr = vsp.get_intrinsics()

        cam_K = [
            float(intr.fx), 0.0, float(intr.ppx),
            0.0, float(intr.fy), float(intr.ppy),
            0.0, 0.0, 1.0
        ]

        # Images
        rgb = np.asanyarray(color_frame.get_data()).astype(np.uint8)

        # ---- CRITICAL FIX: convert depth to millimeters ----
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        depth_mm = np.round(depth_raw.astype(np.float32) * depth_scale_rs * 1000.0).astype(np.uint16)

        # Save
        cv2.imwrite(str(out_dir / args.rgb_name), rgb)
        cv2.imwrite(str(out_dir / args.depth_name), depth_mm)

        # ---- BOP-compatible camera.json ----
        cam_path = out_dir / args.cam_name
        cam_path.write_text(json.dumps({
            "cam_K": cam_K,
            "depth_scale": 0.001   # meters per millimeter (BOP standard)
        }, indent=2))

        print(f"Saved rgb.png, depth.png (mm), camera.json")
        print(f"Depth converted using RealSense scale: {depth_scale_rs}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
