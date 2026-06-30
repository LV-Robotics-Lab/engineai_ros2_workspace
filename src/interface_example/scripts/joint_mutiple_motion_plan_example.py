#!/usr/bin/env python3
"""
Multiple Joint Motion Plan Client Example
Use ROS2 Service to control multiple joint motion planning task queue execution
Load task queue from configuration file and execute each motion planning sequentially

Supported request types:
1. Normal motion planning (default): Specify target joint positions and parameters
2. RESET: Reset to default pose (add request_type: RESET in configuration)
"""

import argparse
import time
from collections import deque
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from interface_protocol.msg import (
    JointMotionPlanRequest,  # type: ignore
    JointMotionPlanState,  # type: ignore
)


class MultipleJointMotionPlanClient(Node):
    """Multi-task joint motion planning Service Client node"""

    def __init__(self, config_path: str):
        super().__init__("multiple_joint_motion_plan_client")

        # Create request topic publisher with Reliable QoS
        request_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._request_publisher = self.create_publisher(
            JointMotionPlanRequest, "/motion/joint_motion_plan/request", request_qos
        )

        # Create state subscriber, matching server state publishing QoS (must be consistent if server is best effort)
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._state_sub = self.create_subscription(
            JointMotionPlanState,
            "/motion/joint_motion_plan/state",
            self.state_callback,
            state_qos,
        )

        # State tracking
        self._current_state = None
        self._previous_state = None
        self._current_request_id = None  # Current request_id from state message
        self._request_sent = False
        self._all_tasks_done = False
        self._should_exit = False  # Flag indicating whether the program should exit

        # Task queue and configuration
        self._motion_queue = deque()
        self._joint_indices = []
        self._current_task = None
        self._total_tasks = 0
        self._completed_tasks = 0

        # Load configuration file
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        """
        Load motion planning task queue from YAML configuration file

        Args:
            config_path: Full path to the configuration file
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if "motion_plan" not in config:
                self.get_logger().error(
                    "Configuration file format error: missing 'motion_plan' field"
                )
                raise ValueError("Invalid config format")

            motion_plan = config["motion_plan"]

            if not motion_plan or len(motion_plan) == 0:
                self.get_logger().error(
                    "No motion planning tasks in configuration file"
                )
                raise ValueError("No motion tasks in config")

            # The first element should contain joint_indices
            if "joint_indices" in motion_plan[0]:
                self._joint_indices = motion_plan[0]["joint_indices"]
                self.get_logger().info(f"Loaded joint indices: {self._joint_indices}")
                motion_tasks = motion_plan[1:]  # The rest are actual motion tasks
            else:
                self.get_logger().error(
                    "Configuration file format error: missing 'joint_indices'"
                )
                raise ValueError("Missing joint_indices")

            # Parse each motion task and add to queue
            for idx, motion_item in enumerate(motion_tasks):
                if "motion" not in motion_item:
                    self.get_logger().warn(
                        f"Skipping task {idx + 1}: missing 'motion' field"
                    )
                    continue

                # Check if it's a RESET type request
                request_type_str = motion_item.get("request_type", "").upper()
                if request_type_str == "RESET":
                    # RESET type task, use default values
                    task = {
                        "name": motion_item.get("motion", f"motion_{idx}"),
                        "request_type": "RESET",
                        "target_positions": [],
                        "duration": 0.0,
                        "use_gravity_compensation": False,
                        "stiffness": [],
                        "damping": [],
                    }
                    self._motion_queue.append(task)
                    self.get_logger().info(
                        f"Loaded task {len(self._motion_queue)}: {task['name']} (RESET)"
                    )
                    continue

                # Normal motion planning task
                # Ensure all values in target_positions are floats
                target_positions = motion_item.get("target_positions", [])
                target_positions = [float(pos) for pos in target_positions]

                task = {
                    "name": motion_item.get("motion", f"motion_{idx}"),
                    "request_type": "PLAN_EXECUTE",
                    "target_positions": target_positions,
                    "duration": float(motion_item.get("duration", 2.0)),
                    "use_gravity_compensation": motion_item.get(
                        "use_gravity_compensation", True
                    ),
                    "stiffness": self._flatten_stiffness_damping(
                        motion_item.get("stiffness", [])
                    ),
                    "damping": self._flatten_stiffness_damping(
                        motion_item.get("damping", [])
                    ),
                }

                # Validate task data
                if len(task["target_positions"]) != len(self._joint_indices):
                    self.get_logger().warn(
                        f"Task '{task['name']}' target position count ({len(task['target_positions'])}) "
                        f"does not match joint index count ({len(self._joint_indices)}), skipping"
                    )
                    continue

                self._motion_queue.append(task)
                self.get_logger().info(
                    f"Loaded task {len(self._motion_queue)}: {task['name']}"
                )

            self._total_tasks = len(self._motion_queue)

            if self._total_tasks == 0:
                self.get_logger().error("No valid motion planning tasks")
                raise ValueError("No valid motion tasks")

            self.get_logger().info("=" * 60)
            self.get_logger().info(
                f"Successfully loaded configuration file: {config_path}"
            )
            self.get_logger().info(f"Total tasks: {self._total_tasks}")
            self.get_logger().info(f"Joint count: {len(self._joint_indices)}")
            self.get_logger().info("=" * 60)

        except FileNotFoundError:
            self.get_logger().error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.get_logger().error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"Failed to load configuration file: {e}")
            raise

    def _flatten_stiffness_damping(self, nested_list):
        """
        Flatten nested stiffness/damping list to a one-dimensional list and ensure all values are floats

        Args:
            nested_list: Nested list (may be a two-dimensional array)

        Returns:
            Flattened one-dimensional list with all elements as floats
        """
        if not nested_list:
            return []

        flattened = []
        for item in nested_list:
            if isinstance(item, list):
                # Convert each element in the list to float
                flattened.extend([float(x) for x in item])
            else:
                # Convert single element to float
                flattened.append(float(item))
        return flattened

    def state_callback(self, msg: JointMotionPlanState):
        """
        State callback, listen to planner status and process task queue

        Args:
            msg: Planner status message
        """
        self._previous_state = self._current_state
        self._current_state = msg.status
        self._current_request_id = msg.request_id  # Save current request_id from state

        # Print status information
        status_names = {
            JointMotionPlanState.STATUS_DISABLED: "DISABLED",
            JointMotionPlanState.IDLE: "IDLE",
            JointMotionPlanState.EXECUTING: "EXECUTING",
            JointMotionPlanState.EXITING: "EXITING",
        }
        status_name = status_names.get(msg.status, f"UNKNOWN({msg.status})")

        task_info = ""
        if self._current_task:
            task_info = f" | Current task: {self._current_task['name']}"

        self.get_logger().info(
            f"Planner status: {status_name} | "
            f"Progress: {msg.progress:.2%} | "
            f"Completed: {self._completed_tasks}/{self._total_tasks}{task_info}"
        )

        # If exit has been requested, return directly
        if self._should_exit:
            return

        # Non-IDLE/EXECUTING status, don't send request, just prompt waiting
        if msg.status not in (
            JointMotionPlanState.IDLE,
            JointMotionPlanState.EXECUTING,
        ):
            if not self._all_tasks_done:
                self.get_logger().info("Planner not ready, waiting...")
            return

        # If status changes from EXECUTING to IDLE, and request has been sent, current task execution is complete
        if (
            self._previous_state == JointMotionPlanState.EXECUTING
            and msg.status == JointMotionPlanState.IDLE
            and self._request_sent
            and self._current_task is not None
        ):
            self._completed_tasks += 1
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"✓ Task completed: {self._current_task['name']}")
            self.get_logger().info(
                f"  Progress: {self._completed_tasks}/{self._total_tasks}"
            )
            self.get_logger().info("=" * 60)

            self._request_sent = False
            self._current_task = None

            # After task completion, if queue still has tasks, continue sending next task
            # Note: Don't send directly here, let the IDLE check logic below handle it
            # This ensures consistency of state transitions

        # If status is IDLE and no request is being sent, send next task
        if msg.status == JointMotionPlanState.IDLE and not self._request_sent:
            if len(self._motion_queue) > 0:
                # Get next task from queue
                next_task = self._motion_queue.popleft()
                self._current_task = next_task
                self._send_next_task(next_task)
            elif self._current_task is None and not self._all_tasks_done:
                # Queue is empty, all tasks completed
                self.get_logger().info("=" * 60)
                self.get_logger().info("🎉 All motion planning tasks completed!")
                self.get_logger().info(f"  Total tasks: {self._total_tasks}")
                self.get_logger().info(f"  Completed: {self._completed_tasks}")
                self.get_logger().info("=" * 60)
                self._all_tasks_done = True

    def _send_next_task(self, task: dict):
        """
        Send the next task in the queue

        Args:
            task: Dictionary containing task parameters
        """
        # Check if request_id is available
        if self._current_request_id is None:
            self.get_logger().error("=" * 60)
            self.get_logger().error(
                "✗ Cannot send request: request_id not yet received from state"
            )
            self.get_logger().error(f"  Failed task: {task['name']}")
            self.get_logger().error(
                f"  Task progress: {self._completed_tasks + 1}/{self._total_tasks}"
            )
            self.get_logger().error("=" * 60)
            self._should_exit = True
            return

        # Create request message
        request_msg = JointMotionPlanRequest()

        # Set request_id (current_request_id + 1)
        request_msg.request_id = self._current_request_id + 1

        # Set request type based on task type
        if task.get("request_type") == "RESET":
            # RESET request: reset to default pose
            request_msg.request_type = JointMotionPlanRequest.REQUEST_RESET
            request_msg.use_gravity_compensation = False
            request_msg.joint_indices = []  # Empty array means reset to default position
            request_msg.target_positions = []
            request_msg.target_velocities = []
            request_msg.execution_time = 0.0
            request_msg.stiffness = []
            request_msg.damping = []

            self.get_logger().info("=" * 60)
            self.get_logger().info(f"▶ Sending reset request: {task['name']}")
            self.get_logger().info(f"  Request ID: {request_msg.request_id}")
            self.get_logger().info("  Request type: RESET (reset to default pose)")
            self.get_logger().info(
                f"  Task progress: {self._completed_tasks + 1}/{self._total_tasks}"
            )
            self.get_logger().info("=" * 60)

        else:
            # Normal motion planning request
            request_msg.request_type = JointMotionPlanRequest.REQUEST_PLAN_EXECUTE

            # Set motion parameters
            request_msg.use_gravity_compensation = task["use_gravity_compensation"]
            request_msg.joint_indices = self._joint_indices
            request_msg.target_positions = task["target_positions"]
            request_msg.execution_time = task["duration"]

            # Set optional parameters
            request_msg.target_velocities = []  # Empty array means auto planning

            if task["stiffness"]:
                request_msg.stiffness = task["stiffness"]
            else:
                request_msg.stiffness = []  # Empty array means use default value

            if task["damping"]:
                request_msg.damping = task["damping"]
            else:
                request_msg.damping = []  # Empty array means use default value

            self.get_logger().info("=" * 60)
            self.get_logger().info(f"▶ Sending motion planning task: {task['name']}")
            self.get_logger().info(f"  Request ID: {request_msg.request_id}")
            self.get_logger().info(
                f"  Task progress: {self._completed_tasks + 1}/{self._total_tasks}"
            )
            self.get_logger().info(f"  Joint indices: {self._joint_indices}")
            self.get_logger().info(
                f"  Target positions: {task['target_positions']} (rad)"
            )
            self.get_logger().info(f"  Execution time: {task['duration']} seconds")
            self.get_logger().info(
                f"  Gravity compensation: {task['use_gravity_compensation']}"
            )
            if task["stiffness"]:
                self.get_logger().info(f"  Stiffness: {task['stiffness']}")
            if task["damping"]:
                self.get_logger().info(f"  Damping: {task['damping']}")
            self.get_logger().info("=" * 60)

        # Publish request to topic
        self.get_logger().info("Publishing request to topic...")
        try:
            self._request_publisher.publish(request_msg)
            self._request_sent = True
            self.get_logger().info(
                "Request published successfully, waiting for status update..."
            )
        except Exception as e:
            # Publish exception, print log and exit program
            self.get_logger().error("=" * 60)
            self.get_logger().error(f"✗ Topic publish exception: {e}")
            self.get_logger().error(f"  Exception task: {task['name']}")
            self.get_logger().error(
                f"  Task progress: {self._completed_tasks + 1}/{self._total_tasks}"
            )
            self.get_logger().error("=" * 60)
            self._should_exit = True
            self._request_sent = False


def main(args=None):
    """Main function"""
    default_config = (
        Path(__file__).resolve().parent
        / ".."
        / "config"
        / "pm01"
        / "motion_plan_wave_hands.yaml"
    ).resolve()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Multiple joint motion planning client - Load and execute motion planning task queue from configuration file"
    )

    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        default=str(default_config),
        help="Full path to motion planning configuration file (YAML format)",
    )

    # Only parse known arguments, ignore ROS2 arguments
    parsed_args, unknown = parser.parse_known_args()
    config_path = parsed_args.config_file

    # Initialize ROS2
    rclpy.init(args=args)

    try:
        # Create topic client node
        client = MultipleJointMotionPlanClient(config_path)
    except Exception as e:
        print(f"Error: Failed to initialize node - {e}")
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    # Wait for state topic to have messages (ensure planner is running and get initial request_id)
    client.get_logger().info(
        "Waiting for planner status message to get initial request_id..."
    )
    deadline = time.monotonic() + 10.0
    try:
        while rclpy.ok() and (
            client._current_state is None
            or client._current_request_id is None
            or client._request_publisher.get_subscription_count() == 0
        ):
            if time.monotonic() > deadline:
                client.get_logger().error("DDS discovery timeout (10s)")
                client.destroy_node()
                rclpy.shutdown()
                return 1
            rclpy.spin_once(client, timeout_sec=0.1)
            client.get_logger().info(f"waiting for DDS discovery: {time.monotonic() - deadline:.2f}s")
    except KeyboardInterrupt:
        client.get_logger().info("User interrupted")
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return 0

    # Check if the initial state is IDLE, if not, exit
    if client._current_state != JointMotionPlanState.IDLE:
        client._should_exit = True
        client.get_logger().error(
            "Initial planner state is not IDLE, cannot start tasks"
        )

    # If current state is IDLE and queue is not empty, immediately trigger first task sending
    # Otherwise wait for state_callback to automatically trigger when IDLE
    if (
        client._current_state == JointMotionPlanState.IDLE
        and not client._request_sent
        and len(client._motion_queue) > 0
        and client._current_request_id is not None
    ):
        client.get_logger().info(
            f"✓ Received planner status message, initial request_id: {client._current_request_id}, ready to start executing tasks"
        )

        next_task = client._motion_queue.popleft()
        client._current_task = next_task
        client._send_next_task(next_task)

    # Run node until all tasks complete or error occurs
    # Note: Subsequent task execution is completely automatically triggered by state_callback when receiving IDLE status
    # This avoids ROS2 Python threading issues
    try:
        while rclpy.ok() and not client._all_tasks_done and not client._should_exit:
            rclpy.spin_once(client, timeout_sec=0.1)
    except KeyboardInterrupt:
        client.get_logger().info("User interrupted")
    finally:
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # If error occurred, exit program and return error code
    if client._should_exit:
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
