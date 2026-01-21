#!/usr/bin/env python3
"""
RealSense D455 Setup and Calibration Script

This script helps set up and calibrate RealSense D455 camera for SAM-6D.
It captures camera intrinsics and saves them in the correct format.

Usage:
    python realsense_setup.py --output_dir Data/myObject/masterChef
"""

import argparse
import json
import numpy as np
import pyrealsense2 as rs
import cv2


def get_realsense_intrinsics():
    """Get camera intrinsics from RealSense D455"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    print("Starting RealSense camera...")
    profile = pipeline.start(config)
    
    try:
        # Get camera intrinsics
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        
        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()
        
        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        print("\n=== RealSense D455 Camera Information ===")
        print(f"\nColor Stream:")
        print(f"  Resolution: {color_intrinsics.width}x{color_intrinsics.height}")
        print(f"  Focal Length: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
        print(f"  Principal Point: cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
        print(f"  Distortion Model: {color_intrinsics.model}")
        
        print(f"\nDepth Stream:")
        print(f"  Resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")
        print(f"  Focal Length: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
        print(f"  Principal Point: cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}")
        print(f"  Depth Scale: {depth_scale} (meters per unit)")
        
        # Create camera.json format
        # Note: depth_scale is multiplied by 1000 to convert from meters to mm
        camera_info = {
            "cam_K": [
                color_intrinsics.fx, 0.0, color_intrinsics.ppx,
                0.0, color_intrinsics.fy, color_intrinsics.ppy,
                0.0, 0.0, 1.0
            ],
            "depth_scale": depth_scale * 1000.0  # Convert m to mm
        }
        
        return camera_info, color_intrinsics, depth_intrinsics
        
    finally:
        pipeline.stop()


def save_camera_json(camera_info, output_path):
    """Save camera intrinsics to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(camera_info, f, indent=2)
    print(f"\nCamera intrinsics saved to: {output_path}")


def test_camera_stream():
    """Test camera stream and show preview"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    # Create align object
    align_to = rs.align(rs.stream.color)
    
    print("\n=== Camera Stream Test ===")
    print("Press 'q' to quit")
    
    try:
        for i in range(100):  # Show 100 frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap to depth
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Stack images
            images = np.hstack((color_image, depth_colormap))
            
            cv2.imshow('RealSense D455 - Press Q to quit', images)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='RealSense D455 Setup and Calibration')
    parser.add_argument('--output_dir', type=str, help='Output directory to save camera.json (e.g., Data/myObject/masterChef)')
    parser.add_argument('--test_stream', action='store_true', help='Test camera stream with preview')
    args = parser.parse_args()
    
    if args.test_stream:
        test_camera_stream()
        return
    
    # Get intrinsics
    camera_info, color_intrinsics, depth_intrinsics = get_realsense_intrinsics()
    
    # Save to file if output directory specified
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'camera.json')
        save_camera_json(camera_info, output_path)
    else:
        # Just print the JSON
        print("\n=== Camera JSON ===")
        print(json.dumps(camera_info, indent=2))
        print("\nSave this to your object directory as 'camera.json'")


if __name__ == '__main__':
    main()
