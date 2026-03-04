#!/usr/bin/env python3
"""
Upper Joint Interpolation Executor
- QuinticSplineInterpolator: Responsible for quintic polynomial interpolation
- JointTrajectoryPlanner: Responsible for discretizing a trajectory at a given frequency
- JointTrajectoryPublisher: ROS2 node that publishes the trajectory back and forth at a fixed frequency
"""

import sys
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from interface_protocol.msg import MotionState
from interface_protocol.msg import JointOverrideCommand  # type: ignore


class QuinticSplineInterpolator:
    """Quintic polynomial interpolation q(t) = a0 + a1 t + ... + a5 t^5"""

    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, duration: float):
        """
        Initialize the quintic spline interpolator

        Args:
            start_pos: Starting joint positions (n,)
            end_pos: Target joint positions (n,)
            duration: Interpolation duration (seconds)
        """

        self.start_pos = np.asarray(start_pos, dtype=float)
        self.end_pos = np.asarray(end_pos, dtype=float)
        self.duration = float(duration)
        self._compute_coefficients()

    def _compute_coefficients(self):
        """Compute quintic polynomial coefficients q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5"""

        T = self.duration
        q0 = self.start_pos
        qf = self.end_pos

        v0 = vf = a0 = af = np.zeros_like(q0)

        self.a0 = q0
        self.a1 = v0
        self.a2 = a0 / 2.0

        dq = qf - q0
        self.a3 = (20 * dq - (8 * vf + 12 * v0) * T - (3 * a0 - af) * T**2) / (2 * T**3)
        self.a4 = (
            30 * (q0 - qf) + (14 * vf + 16 * v0) * T + (3 * a0 - 2 * af) * T**2
        ) / (2 * T**4)
        self.a5 = (12 * (qf - q0) - 6 * (vf + v0) * T - (a0 - af) * T**2) / (2 * T**5)

    def interpolate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate at a given time point

        Args:
            t: Current time (0 <= t <= duration)

        Returns:
            position: Joint positions
            velocity: Joint velocities
            acceleration: Joint accelerations
        """

        t = np.clip(t, 0.0, self.duration)
        pos = (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )
        vel = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t**2
            + 4 * self.a4 * t**3
            + 5 * self.a5 * t**4
        )
        return pos, vel


class JointTrajectoryPlanner:
    """Given start/end positions and duration, sample a trajectory at a given frequency"""

    def __init__(self, frequency: float = 100.0):
        """
        Initialize the joint trajectory planner

        Args:
            frequency: Sampling frequency (Hz)
        """

        self.frequency = float(frequency)

    def plan(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        duration: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a forward trajectory, return [(pos, vel), ...]

        Args:
            start_pos: Starting joint positions (n,)
            end_pos: Target joint positions (n,)
            duration: Trajectory duration (seconds)

        Returns:
            List of tuples containing joint positions and velocities
        """

        interpolator = QuinticSplineInterpolator(start_pos, end_pos, duration)
        steps = int(duration * self.frequency)
        # np.linspace includes both ends, here we only need the discrete length
        times = np.linspace(0.0, duration, steps, endpoint=True)
        traj: List[Tuple[np.ndarray, np.ndarray]] = []
        for t in times:
            traj.append(interpolator.interpolate(t))
        return traj


class JointOverridePublisher(Node):
    """Given a trajectory array, publish it back and forth at a fixed frequency"""

    def __init__(
        self,
        joint_indices: List[int],
        velocity: np.ndarray,
        feed_forward_torque: np.ndarray,
        torque: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        trajectory: List[Tuple[np.ndarray, np.ndarray]],
        freq: float = 100.0,
    ):
        """
        Initialize the joint override publisher node

        Args:
            joint_indices: List of joint indices to control
            velocity: Joint velocities to publish
            feed_forward_torque: Feed forward torques to publish
            torque: Joint torques to publish
            stiffness: Joint stiffness to publish
            damping: Joint damping to publish
            trajectory: List of tuples containing joint positions and velocities
            freq: Publishing frequency (Hz)
        """

        super().__init__("joint_override_publisher")

        self.should_exit = False
        self.joint_indices = joint_indices
        self.velocity = velocity
        self.feed_forward_torque = feed_forward_torque
        self.torque = torque
        self.stiffness = stiffness
        self.damping = damping

        self.freq = float(freq)
        self.trajectory = trajectory

        self.idx = 0
        self.direction = 1  # 1: forward, -1: backward

        self.publisher_ = self.create_publisher(
            JointOverrideCommand,
            "/motion/joint_override_command",
            10,
        )

        self.motion_state_subscriber = self.create_subscription(
            MotionState,
            "/motion/motion_state",
            self.motion_state_callback,
            10,
        )

        self.timer = self.create_timer(1.0 / self.freq, self.timer_callback)

        self.get_logger().info(
            f"joint_override_publisher started: len={len(self.trajectory)}, freq={self.freq}Hz"
        )

    def motion_state_callback(self, msg: MotionState):
        if msg.current_motion_task != "lower_body_balance":
            self.get_logger().error("Not in lower_body_balance state, exiting")
            self.should_exit = True

    def timer_callback(self):
        """Timer callback to publish the next command in the trajectory"""

        pos, _ = self.trajectory[self.idx]
        self._publish_command(pos)

        self.idx += self.direction
        if self.idx >= len(self.trajectory) - 1:
            self.idx = len(self.trajectory) - 1
            self.direction = -1
        elif self.idx <= 0:
            self.idx = 0
            self.direction = 1

    def _publish_command(self, pos: np.ndarray):
        """
        Publish a joint override command message with the given position and velocity

        Args:
            pos: Joint positions to publish
        """

        msg = JointOverrideCommand()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""

        msg.weight = 1.0
        msg.joint_indices = self.joint_indices
        msg.position = pos.tolist()
        msg.velocity = self.velocity.tolist()
        msg.feed_forward_torque = self.feed_forward_torque.tolist()
        msg.torque = self.torque.tolist()
        msg.stiffness = self.stiffness.tolist()
        msg.damping = self.damping.tolist()

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    # 1) First, plan a trajectory
    timer_freq = 100.0
    start_pos = np.array([0.0, 0.0])

    joint_indices = [14, 19]
    end_pos = np.array([1.57, -1.57])
    velocity = np.array([0.0, 0.0])
    feed_forward_torque = np.array([0.0, 0.0])
    torque = np.array([0.0, 0.0])
    stiffness = np.array([20.0, 20.0])
    damping = np.array([1.0, 1.0])

    planner = JointTrajectoryPlanner(frequency=timer_freq)
    trajectory = planner.plan(start_pos, end_pos, duration=5.0)

    # 2) Then, play it back and forth
    node = JointOverridePublisher(
        freq=timer_freq,
        trajectory=trajectory,
        joint_indices=joint_indices,
        velocity=velocity,
        feed_forward_torque=feed_forward_torque,
        torque=torque,
        stiffness=stiffness,
        damping=damping,
    )

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("User interrupted")
    except Exception as e:
        node.get_logger().error(f"Error occurred: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
