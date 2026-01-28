#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# Hyperparameters
OFFSET = np.array([0.034, -0.040, 0.09], dtype=np.float64)        # meters

# Helper Functions
def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
    q /= np.linalg.norm(q) + 1e-12
    return q  # wxyz

# Class Definition
class ExtrinsicsBroadcaster(Node):
    """ Broadcasts static TF from camera to robot base using extrinsics from .npz file. """
    def __init__(self):
        super().__init__("camera_extrinsics_broadcaster")
        # Parameters
        self.declare_parameter("npz_path","/home/nikolaraicevic/Workspace/Internal/Camera_Calibration/src/Xarm6_Camera_Calibration/camera_extrinsics_036322250488.npz")
        self.declare_parameter("parent_frame", "robot_base")
        self.declare_parameter("child_frame", "camera_color_optical_frame")
        self.declare_parameter("key", "cam2arm")

        # Get parameters
        npz_path = self.get_parameter("npz_path").value
        parent = self.get_parameter("parent_frame").value
        child = self.get_parameter("child_frame").value
        key = self.get_parameter("key").value

        # Obtain rotation and translation from .npz in cam2arm frame
        cam2arm = np.load(npz_path)[key]  
        R = cam2arm[:3, :3].astype(np.float64)
        t = cam2arm[:3, 3].astype(np.float64) + OFFSET
        print("Loaded extrinsics:")
        print("Rotation:\n", R)
        print("Translation:\n", t)
        print(f"cam2arm shape: {cam2arm.shape}")
        print(f"cam2arm:\n{cam2arm}")

        q = rotmat_to_quat_wxyz(R)
        self.broadcaster = StaticTransformBroadcaster(self)
        tfm = TransformStamped()
        tfm.header.stamp = self.get_clock().now().to_msg()
        tfm.header.frame_id = parent
        tfm.child_frame_id = child
        tfm.transform.translation.x = float(t[0])
        tfm.transform.translation.y = float(t[1])
        tfm.transform.translation.z = float(t[2])
        tfm.transform.rotation.w = float(q[0])
        tfm.transform.rotation.x = float(q[1])
        tfm.transform.rotation.y = float(q[2])
        tfm.transform.rotation.z = float(q[3])
        self.broadcaster.sendTransform(tfm)


def main():
    rclpy.init()
    node = ExtrinsicsBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
