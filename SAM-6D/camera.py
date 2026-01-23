#!/usr/bin/env python3

import pyrealsense2 as rs
from pathlib import Path
import numpy as np
import argparse
import json
import cv2

# Hyperparameters
W = 640                         # Width
H = 480                         # Height
FPS = 30                        # Frames per second
WARMUP = 30                     # Number of warmup frames
DEPTH_SCALE = 1.0               # Scale to convert depth to mm for BOP storage
INTRINSICS_FOR = "color"        # "color" or "depth" 

RGB_NAME = "rgb.png"            # RGB image filename
DEPTH_NAME = "depth.png"        # Depth image filename
CAMINFO_NAME = "camera.json"    # Camera intrinsics filename

# RealSenseCamera class
class RealSenseCamera:
    """
    Thin wrapper around RealSense pipeline + device enumeration.
    """

    def __init__(
        self,
        width=W,
        height=H,
        fps=FPS,
        warmup=WARMUP,
        depth_scale=DEPTH_SCALE,
        intrinsics_for=INTRINSICS_FOR,
        rgb_name=RGB_NAME,
        depth_name=DEPTH_NAME,
        caminfo_name=CAMINFO_NAME,
        out_dir="./CameraOutputs",
        align_to_color: bool = True,
    ):
        """ Initializes RealSense camera and starts pipeline """
        # Store parameters
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup = int(warmup)

        self.depth_scale = float(depth_scale)     # BOP depth_scale
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be > 0")

        self.intrinsics_for = intrinsics_for
        self.rgb_name = rgb_name
        self.depth_name = depth_name
        self.caminfo_name = caminfo_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.align_to_color = bool(align_to_color)

        # Device enumeration
        self.ctx = rs.context()
        self.devices = self.ctx.query_devices()
        if len(self.devices) == 0:
            raise RuntimeError("No RealSense device connected.")

        # Pipeline config
        self.pipeline = rs.pipeline(self.ctx)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(self.config)

        # RealSense depth scale: meters per raw unit
        self.depth_scale_rs_m_per_unit = float(self.profile.get_device().first_depth_sensor().get_depth_scale())

        # Alignment object
        self.align = rs.align(rs.stream.color) if self.align_to_color else None

        # Depth post-processing filters
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # Prevent decimation from changing resolution
        try:
            self.decimation.set_option(rs.option.filter_magnitude, 1.0)
        except Exception:
            pass

        # Warmup
        for _ in range(self.warmup):
            self.pipeline.wait_for_frames()

    def __del__(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    def get_camera_info(self, print_info: bool = True):
        """
        Returns list of dictionaries with device info.
        """
        infos = []

        def safe_info(dev, info_enum):
            try:
                return dev.get_info(info_enum) if dev.supports(info_enum) else None
            except Exception:
                return None

        def safe_option_range(sensor, opt_enum):
            try:
                if sensor.supports(opt_enum):
                    r = sensor.get_option_range(opt_enum)
                    return {"min": float(r.min), "max": float(r.max), "step": float(r.step), "default": float(r.default)}
            except Exception:
                pass
            return None

        # Use the active streaming device's depth sensor (pipeline-selected device)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        max_distance_range = safe_option_range(depth_sensor, rs.option.max_distance)  # often None on D455

        for idx, dev in enumerate(self.devices):
            info = {
                "index": idx,
                "name": safe_info(dev, rs.camera_info.name),
                "serial": safe_info(dev, rs.camera_info.serial_number),
                "firmware": safe_info(dev, rs.camera_info.firmware_version),
                "product_line": safe_info(dev, rs.camera_info.product_line),
                "usb_type": safe_info(dev, rs.camera_info.usb_type_descriptor),
                "depth_scale": depth_scale,
                "max_distance_range_m": max_distance_range,
            }
            infos.append(info)

            if print_info:
                print(f"\n<==== Device {idx}: {info['name']} ====>")
                print(f"Name           : {info['name']}")
                print(f"Serial Number  : {info['serial']}")
                print(f"Firmware       : {info['firmware']}")
                print(f"Product Line   : {info['product_line']}")
                print(f"USB Type       : {info['usb_type']}")
                print(f"Depth Scale    : {info['depth_scale']:.6f} meters/unit")

                if info["max_distance_range_m"] is None:
                    print("Max Distance   : <not supported>")
                else:
                    r = info["max_distance_range_m"]
                    print(f"Max Distance   : {r['max']:.2f} m  (min={r['min']:.2f}, default={r['default']:.2f})")

        return infos

    def get_camera_intrinsics( self, save_json: bool = True, print_info: bool = True) -> dict:
        """
        Fetch intrinsics for the active stream profile and (optionally) save BOP-style camera.json.

        Conventions:
        - cam_K is 3x3 row-major flattened.
        - depth_scale should match how you store depth.png.
            If depth.png is uint16 millimeters, depth_scale = 1.0.

        Returns:
        dict with two sections:
            - "intrinsics": full intrinsics + distortion params
            - "bop": {"cam_K": [...], "depth_scale": ...}
        """
        # Validate
        if self.intrinsics_for not in ("color", "depth"):
            raise ValueError('self.intrinsics_for must be "color" or "depth"')
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be > 0")

        # Select active stream and get intrinsics
        rs_stream = rs.stream.color if self.intrinsics_for == "color" else rs.stream.depth
        vsp = self.profile.get_stream(rs_stream).as_video_stream_profile()
        intr = vsp.get_intrinsics()

        # Parse intrinsics
        fx, fy = float(intr.fx), float(intr.fy)
        cx, cy = float(intr.ppx), float(intr.ppy)
        w, h = int(intr.width), int(intr.height)
        distortion_model = str(intr.model)
        distortion_coeffs = [float(c) for c in intr.coeffs]

        intrinsics = {
            "stream": self.intrinsics_for,
            "width": w,
            "height": h,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion_model": distortion_model,
            "distortion_coeffs": distortion_coeffs,
        }

        cam_K = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]
        bop = {"cam_K": cam_K, "depth_scale": float(self.depth_scale)}

        if print_info:
            print(f"\n<==== Intrinsics: {self.intrinsics_for.upper()} ({w}x{h}) ====>")
            print(f"  fx, fy      : {fx:.6f}, {fy:.6f}")
            print(f"  cx, cy      : {cx:.6f}, {cy:.6f}")
            print(f"  distortion  : {distortion_model}")

            labels = ["k1", "k2", "p1", "p2", "k3"]
            if len(distortion_coeffs) == 5:
                coeff_str = ", ".join(f"{labels[i]}={distortion_coeffs[i]:+.6e}" for i in range(5))
            else:
                coeff_str = ", ".join(f"c{i}={c:+.6e}" for i, c in enumerate(distortion_coeffs))
            print(f"  coeffs      : {coeff_str}")

            print("  cam_K       :")
            print(f"    [{cam_K[0]:.6f}, {cam_K[1]:.1f}, {cam_K[2]:.6f}]")
            print(f"    [{cam_K[3]:.1f}, {cam_K[4]:.6f}, {cam_K[5]:.6f}]")
            print(f"    [{cam_K[6]:.1f}, {cam_K[7]:.1f}, {cam_K[8]:.1f}]")
            print(f"  depth_scale : {bop['depth_scale']}")

        # Save BOP json file
        if save_json:
            caminfo_path = self.out_dir / self.caminfo_name
            caminfo_path.write_text(json.dumps(bop, indent=2))

        return {"intrinsics": intrinsics, "bop": bop}
    
    def frames(self):
        """
        Generator of synchronized frames.

        Yields:
          color_bgr: uint8 HxWx3
          depth_bop: uint16 HxW (BOP stored units; depth_mm = depth_bop / self.depth_scale)
        """
        while True:
            fs = self.pipeline.wait_for_frames()

            if self.align is not None:
                fs = self.align.process(fs)

            depth_frame = fs.get_depth_frame()
            color_frame = fs.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Filter depth
            depth_frame = self.decimation.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)

            color_bgr = np.asanyarray(color_frame.get_data()).astype(np.uint8)
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

            # raw -> mm
            depth_mm = np.round(depth_raw.astype(np.float32) * self.depth_scale_rs_m_per_unit * 1000.0).astype(np.uint16)

            # mm -> BOP stored units
            depth_bop = np.round(depth_mm.astype(np.float32) * self.depth_scale).astype(np.uint16)

            yield color_bgr, depth_bop

    def get_camera_stream( self, window_name: str = "RealSense - (q) quit  (c) capture", max_depth_mm: int = 6000):
        """
        Preview + capture using the same frames() generator.

        Preview is stable grayscale derived from the SAVED depth values (depth_bop).
        """
        if max_depth_mm <= 0:
            raise ValueError("max_depth_mm must be > 0")

        # Fixed visualization scale in BOP units
        max_bop = int(round(max_depth_mm * self.depth_scale))
        max_bop = max(max_bop, 1)

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            for color_bgr, depth_bop in self.frames():
                # Stable preview: map [0..max_bop] -> [0..255]
                depth_u8 = (np.minimum(depth_bop, max_bop).astype(np.float32) * (255.0 / max_bop)).astype(np.uint8)
                depth_gray_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

                cv2.imshow(window_name, np.hstack((color_bgr, depth_gray_bgr)))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    rgb_path = self.out_dir / self.rgb_name
                    depth_path = self.out_dir / self.depth_name

                    cv2.imwrite(str(rgb_path), color_bgr)
                    cv2.imwrite(str(depth_path), depth_bop)

                    print(
                        "Captured:\n"
                        f"  RGB         : {rgb_path}\n"
                        f"  Depth (BOP) : {depth_path} (uint16; depth_mm = value / {self.depth_scale})\n"
                        f"  Align color : {self.align_to_color}\n"
                        f"  Preview max : {max_depth_mm} mm"
                    )
        finally:
            cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser(description="Capture synchronized RGB + depth frames from RealSense camera.")
    ap.add_argument("-o", "--out_dir", type=str, default="./CameraOutputs")
    args = ap.parse_args()

    realsensecamera = RealSenseCamera(out_dir=args.out_dir)
    try:
        realsensecamera.get_camera_info(print_info=True)
        realsensecamera.get_camera_intrinsics(save_json=True, print_info=True)
        realsensecamera.get_camera_stream(window_name="RealSense - (q) quit  (c) capture")
    finally:
        del realsensecamera

if __name__ == "__main__":
    main()
