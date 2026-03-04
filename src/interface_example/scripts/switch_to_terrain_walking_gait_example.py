import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from interface_protocol.msg import MotionState
from interface_protocol.msg import MotionStateRequest


class MotionStateSwitcher(Node):
    def __init__(self, target_motion: str, timer_period: float = 0.1):
        super().__init__("motion_state_switcher")

        # Create request topic publisher with Reliable QoS
        request_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._request_publisher = self.create_publisher(
            MotionStateRequest, "/motion/set_motion_state", request_qos
        )

        # Create state subscriber, matching server state publishing QoS (must be consistent if server is best effort)
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._motion_state_sub = self.create_subscription(
            MotionState,
            "/motion/motion_state",
            self.motion_state_callback,
            state_qos,
        )

        # Internal state variables
        self.should_exit = False
        self._motion_state = ""
        self._available_transition_motions = []
        self._target_motion_state = target_motion
        self._state_switched = False  # Flag to track if we've already switched
        self._publish_timestamp = None  # Track when we published the message
        self._waiting_for_switch = (
            False  # Flag to indicate we're waiting for switch completion
        )
        self._switch_timeout = 3.0  # Timeout in seconds
        self._last_wait_log_time = 0.0  # Track last time we logged waiting status

        # Create timer to periodically check and print status
        self._timer_period = timer_period
        self._timer = self.create_timer(self._timer_period, self.timer_callback)

        self.get_logger().info(
            f"MotionStateSwitcher initialized with target motion: {self._target_motion_state}"
        )

    def motion_state_callback(self, msg: MotionState):
        """
        Callback to cache motion state information.
        This only stores the data without any logging.
        """
        self._motion_state = msg.current_motion_task
        self._available_transition_motions = msg.available_transition_motions

    def timer_callback(self):
        # If we're waiting for the switch to complete
        if self._waiting_for_switch:
            current_time = time.time()
            elapsed_time = current_time - self._publish_timestamp

            # Check if the switch was successful
            if self._motion_state == self._target_motion_state:
                self.get_logger().info(
                    f"Motion state successfully switched to '{self._target_motion_state}' in {elapsed_time:.2f}s"
                )
                self.get_logger().info("Exiting safely...")
                self._timer.cancel()
                self.should_exit = True
                return

            # Check if timeout exceeded
            if elapsed_time >= self._switch_timeout:
                self.get_logger().warn(
                    f"Motion state switch to '{self._target_motion_state}' did not complete within {self._switch_timeout}s timeout"
                )
                self.get_logger().warn(f"Current state is still: {self._motion_state}")
                self.get_logger().warn("Exiting with warning...")
                self._timer.cancel()
                self.should_exit = True
                return

            # Still waiting, show status periodically (every 0.5s)
            if current_time - self._last_wait_log_time >= 0.5:
                self.get_logger().info(
                    f"Waiting for switch to complete... ({elapsed_time:.1f}s/{self._switch_timeout}s)"
                )
                self._last_wait_log_time = current_time
            return

        # Normal operation: print current state
        self.get_logger().info("-------------------------")
        self.get_logger().info(f"Current motion state: {self._motion_state}")
        self.get_logger().info(
            f"Available transition motions: {self._available_transition_motions}"
        )
        self.get_logger().info("-------------------------")

        # Check if we have valid state data
        if self._motion_state == "" or len(self._available_transition_motions) == 0:
            self.get_logger().warn("Motion state data not available yet, waiting...")
            return

        # Check if we've already switched
        if self._state_switched:
            self.get_logger().error(
                "Unexpected state: already switched but not waiting"
            )
            self._timer.cancel()
            self.should_exit = True
            return

        # Check if target motion is in available transitions
        if self._target_motion_state not in self._available_transition_motions:
            self.get_logger().error(
                f"Motion '{self._target_motion_state}' not available in current transitions"
            )
            self._timer.cancel()
            self.should_exit = True
            return

        # Both conditions met, switch to target motion
        self.get_logger().info(
            f"Motion '{self._target_motion_state}' is available, switching now..."
        )
        self.set_motion_state(self._target_motion_state)
        self._state_switched = True
        self._waiting_for_switch = True
        self._publish_timestamp = time.time()

    def set_motion_state(self, motion_state: str):
        """Publish motion state change request"""
        msg = MotionStateRequest()
        msg.target_motion_name = motion_state
        self._request_publisher.publish(msg)
        self.get_logger().info(f"Published motion state request: {motion_state}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Motion state switcher for transitioning between motion states"
    )
    parser.add_argument(
        "--target-motion",
        "-m",
        type=str,
        default="rl_terrain",
        help="Target motion state to switch to",
    )
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    switcher = MotionStateSwitcher(parsed_args.target_motion)

    try:
        while rclpy.ok() and not switcher.should_exit:
            rclpy.spin_once(switcher, timeout_sec=0.1)
    except KeyboardInterrupt:
        switcher.get_logger().info("Interrupted by user")
    finally:
        switcher.get_logger().info("Shutting down...")
        switcher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
