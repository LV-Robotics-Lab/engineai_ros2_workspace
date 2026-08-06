#!/usr/bin/env python3
"""
t800pro dexterous-hand example (joint_bridge mode)

Publishes full-body JointCommand on /hardware/joint_command:
  - Selected hand joints: linear interpolation to the target pose
  - Remaining joints: hold the positions captured on entry

Poses:
  extend     selected hand(s) fully extended
  loose_fist selected hand(s) in a loose fist
  both       loose_fist -> hold -> extend

Hands:
  both   left + right (default)
  left   left hand only
  right  right hand only

Prerequisites:
  1. Robot or simulation is running
  2. Current motion is joint_bridge (typically entered from pd_stand)
  3. Robot is standing stably

Usage:
  source /opt/ros/humble/setup.bash
  source scripts/setup_host_ros_env.bash
  source build/x86_64/_install/share/interface_protocol/local_setup.sh

  # Both hands
  python3 loose_fist_example.py --pose extend
  python3 loose_fist_example.py --pose loose_fist
  python3 loose_fist_example.py --pose both

  # Left hand only
  python3 loose_fist_example.py --hand left --pose loose_fist
  python3 loose_fist_example.py --hand left --pose extend
  python3 loose_fist_example.py --hand left --pose both

  # Right hand only
  python3 loose_fist_example.py --hand right --pose loose_fist
  python3 loose_fist_example.py --hand right --pose extend
  python3 loose_fist_example.py --hand right --pose both

  # With parameters
  python3 loose_fist_example.py --hand left --pose both --exec-time 3.0 --hold-time 2.5 --rate 500.0
  python3 loose_fist_example.py --hand right --pose loose_fist --exec-time 2.0 --hold-time 1.0 --rate 200.0
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from interface_protocol.msg import JointCommand, JointState, MotionState

COMMAND_TOPIC = "/hardware/joint_command"
STATE_TOPIC = "/hardware/joint_state"
MOTION_TOPIC = "/motion/motion_state"
REQUIRED_MOTION = "joint_bridge"

NUM_JOINTS = 43  # t800pro: J00..J42
LEFT_HAND_JOINTS = list(range(20, 27))
RIGHT_HAND_JOINTS = list(range(34, 41))

# Loose fist: stay clear of hard limits; proximal bend > distal bend
LEFT_POSES = {
    "extend": [0.0] * 7,
    "loose_fist": [-0.85, 0.62, 0.48, -0.80, -0.58, -0.80, -0.58],
}
RIGHT_POSES = {
    "extend": [0.0] * 7,
    "loose_fist": [0.85, 0.62, 0.48, 0.80, 0.58, 0.80, 0.58],
}

STIFFNESS = [float(x) for x in (
    [1080, 480, 880, 1000, 800, 100]
    + [1080, 480, 880, 1000, 800, 100]
    + [200]
    + [120, 120, 120, 120, 120, 10, 10]
    + [5.0] * 7
    + [120, 120, 120, 120, 120, 10, 10]
    + [5.0] * 7
    + [100, 100]
)]
DAMPING = [float(x) for x in (
    [25, 25, 25, 25, 2, 2]
    + [25, 25, 25, 25, 2, 2]
    + [1]
    + [1.8, 1.5, 1.5, 1.8, 1.2, 0.8, 0.8]
    + [0.2] * 7  # left hand (higher damping to reduce jitter)
    + [1.8, 1.5, 1.5, 1.8, 1.2, 0.8, 0.8]
    + [0.2] * 7  # right hand
    + [1, 1]
)]

QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return [ai + (bi - ai) * t for ai, bi in zip(a, b)]


def hand_joints(hand):
    if hand == "left":
        return LEFT_HAND_JOINTS
    if hand == "right":
        return RIGHT_HAND_JOINTS
    return LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS


def goal_hands(pose_name, hand):
    if hand == "left":
        return LEFT_POSES[pose_name]
    if hand == "right":
        return RIGHT_POSES[pose_name]
    return LEFT_POSES[pose_name] + RIGHT_POSES[pose_name]


class HandBridgeExample(Node):
    def __init__(self, rate_hz):
        super().__init__("loose_fist_t800pro_example")
        self._dt = 1.0 / rate_hz
        self._latest_state = None
        self._latest_motion = MotionState()
        self._cmd_pub = self.create_publisher(JointCommand, COMMAND_TOPIC, QOS)
        self.create_subscription(JointState, STATE_TOPIC, self._on_state, QOS)
        self.create_subscription(MotionState, MOTION_TOPIC, self._on_motion, QOS)

        deadline = time.time() + 5.0
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def _on_state(self, msg):
        self._latest_state = msg

    def _on_motion(self, msg):
        self._latest_motion = msg

    def check_motion(self):
        current = self._latest_motion.current_motion_task
        if not current:
            self.get_logger().warning("No message received on %s yet" % MOTION_TOPIC)
            return False
        if current != REQUIRED_MOTION:
            self.get_logger().error(
                "Current motion=[%s], required=[%s]. Switch first, e.g.:\n"
                "  ros2 topic pub --once /motion/set_motion_state "
                "interface_protocol/msg/MotionStateRequest "
                '"{target_motion_name: \'%s\'}"' % (current, REQUIRED_MOTION, REQUIRED_MOTION)
            )
            return False
        self.get_logger().info("Current motion: %s" % current)
        return True

    def wait_joint_state(self, timeout_sec=5.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_state and len(self._latest_state.position) == NUM_JOINTS:
                return list(self._latest_state.position)
        self.get_logger().error(
            "Timed out waiting for %s (need %d joints)" % (STATE_TOPIC, NUM_JOINTS)
        )
        return None

    def _make_command(self, positions):
        msg = JointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = positions
        msg.velocity = [0.0] * NUM_JOINTS
        msg.feed_forward_torque = [0.0] * NUM_JOINTS
        msg.torque = [0.0] * NUM_JOINTS
        msg.stiffness = list(STIFFNESS)
        msg.damping = list(DAMPING)
        msg.parallel_parser_type = 0  # CLASSIC_PARSER
        return msg

    def _publish_for(self, positions, duration):
        deadline = time.time() + duration
        while time.time() < deadline and rclpy.ok():
            self._cmd_pub.publish(self._make_command(positions))
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self._dt)

    def run_pose(self, pose_name, hand, hold_positions, exec_time, hold_time):
        joints = hand_joints(hand)
        start_hands = [hold_positions[i] for i in joints]
        goal = goal_hands(pose_name, hand)
        self.get_logger().info(
            "Running pose [%s] on hand [%s] (%.1fs)..." % (pose_name, hand, exec_time)
        )

        t0 = time.time()
        while rclpy.ok():
            elapsed = time.time() - t0
            if elapsed >= exec_time:
                break
            hands = lerp(start_hands, goal, elapsed / exec_time)
            cmd_pos = list(hold_positions)
            for idx, value in zip(joints, hands):
                cmd_pos[idx] = value
            self._cmd_pub.publish(self._make_command(cmd_pos))
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self._dt)

        final_pos = list(hold_positions)
        for idx, value in zip(joints, goal):
            final_pos[idx] = value
        hold_positions[:] = final_pos
        self._publish_for(final_pos, hold_time)
        self.get_logger().info("Pose [%s] on hand [%s] done" % (pose_name, hand))
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="t800pro dexterous-hand example (joint_bridge)"
    )
    parser.add_argument(
        "--pose",
        choices=["extend", "loose_fist", "both"],
        default="both",
        help="Target pose",
    )
    parser.add_argument(
        "--hand",
        choices=["both", "left", "right"],
        default="both",
        help="Which hand(s) to move; the other hand holds current position",
    )
    parser.add_argument(
        "--exec-time", type=float, default=3.0, help="Interpolation time per pose (s)"
    )
    parser.add_argument(
        "--hold-time", type=float, default=2.5, help="Hold time after reaching pose (s)"
    )
    parser.add_argument(
        "--rate", type=float, default=500.0, help="JointCommand publish rate (Hz)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pose_sequence = ["loose_fist", "extend"] if args.pose == "both" else [args.pose]

    rclpy.init()
    node = HandBridgeExample(rate_hz=args.rate)
    try:
        if not node.check_motion():
            return 1
        hold_positions = node.wait_joint_state()
        if hold_positions is None:
            return 1
        for pose_name in pose_sequence:
            if not node.run_pose(
                pose_name, args.hand, hold_positions, args.exec_time, args.hold_time
            ):
                return 1
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
