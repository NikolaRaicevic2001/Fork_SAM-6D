#!/usr/bin/env python3
import json
import rclpy
import socket

from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped

class Sam6dPosePublisher(Node):
    def __init__(self):
        super().__init__("sam6d_pose_publisher")

        # Parameters
        self.declare_parameter("udp_ip", "127.0.0.1")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("pose_topic", "/sam6d/object_pose")
        self.declare_parameter("parent_frame", "camera_color_optical_frame")
        self.declare_parameter("child_frame", "sam6d_object")
        self.declare_parameter("publish_tf", True)

        udp_ip = self.get_parameter("udp_ip").value
        udp_port = int(self.get_parameter("udp_port").value)
        self.pose_topic = self.get_parameter("pose_topic").value
        self.parent_frame = self.get_parameter("parent_frame").value
        self.child_frame = self.get_parameter("child_frame").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        # ROS pubs
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # UDP socket (non-blocking via timeout + timer)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_ip, udp_port))
        self.sock.setblocking(False)
        self.get_logger().info(f"Listening UDP on {udp_ip}:{udp_port}, publishing {self.pose_topic} and TF={self.publish_tf}")

        self.timer = self.create_timer(0.01, self.poll_udp)  # 100 Hz poll

    def poll_udp(self):
        try:
            data, _ = self.sock.recvfrom(65535)
        except BlockingIOError:
            return

        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception as e:
            self.get_logger().warn(f"Bad UDP packet: {e}")
            return

        # unpack
        t = msg.get("t_m", None)
        q = msg.get("q_wxyz", None)
        frame_id = msg.get("frame_id", self.parent_frame)

        if t is None or q is None or len(t) != 3 or len(q) != 4:
            self.get_logger().warn("Missing/invalid t_m or q_wxyz")
            return

        # ROS time: stamp now 
        stamp = self.get_clock().now().to_msg()

        # PoseStamped
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(t[0])
        ps.pose.position.y = float(t[1])
        ps.pose.position.z = float(t[2])
        ps.pose.orientation.w = float(q[0])
        ps.pose.orientation.x = float(q[1])
        ps.pose.orientation.y = float(q[2])
        ps.pose.orientation.z = float(q[3])
        self.pose_pub.publish(ps)

        # TF
        if self.publish_tf:
            tfm = TransformStamped()
            tfm.header.stamp = stamp
            tfm.header.frame_id = frame_id
            tfm.child_frame_id = self.child_frame
            tfm.transform.translation.x = float(t[0])
            tfm.transform.translation.y = float(t[1])
            tfm.transform.translation.z = float(t[2])
            tfm.transform.rotation.w = float(q[0])
            tfm.transform.rotation.x = float(q[1])
            tfm.transform.rotation.y = float(q[2])
            tfm.transform.rotation.z = float(q[3])
            self.tf_broadcaster.sendTransform(tfm)

def main():
    rclpy.init()
    node = Sam6dPosePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
