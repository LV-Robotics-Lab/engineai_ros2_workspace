#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Header
from interface_protocol.msg import JointCommand, JointState


class JointTestExample(Node):
    def __init__(self):
        super().__init__('joint_test_example')

        # Hardcoded target positions (from joint_test.yaml)
        # Format: [group1, group2, group3, group4, group5, group6]
        self.target_positions = [
            0.0, 0.5, 1.57, 0.6, -0.3, 0,  # Group 1: 6 joints
            -0.0, -0.5, -1.57, 0.6, -0.3, 0,  # Group 2: 6 joints
            0,  # Group 3: 1 joint
            0, 0.3, -0, -0.4, 0,  # Group 4: 5 joints
            0, -0.2, -0, -0.3, 0,  # Group 5: 5 joints
            0  # Group 6: 1 joint
        ]

        # Number of interpolation steps for each joint (200 steps at 100Hz = 2 seconds)
        self.num_steps = [200] * len(self.target_positions)

        # Control parameters
        self.stiffness = 400.0
        self.damping = 5.0

        # State variables
        self.initial_positions = None
        self.current_steps = [0] * len(self.target_positions)
        self.reached_targets = [False] * len(self.target_positions)
        self.interpolated_positions = []
        self.initialized = False
        self.all_reached_logged = False  # Flag to log "all reached" message only once

        # Create QoS profile matching C++ version (best_effort, volatile, depth=3)
        qos_profile = QoSProfile(
            depth=3,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Create subscriber for joint state
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/hardware/joint_state',
            self.joint_state_callback,
            qos_profile
        )

        # Create publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointCommand,
            '/hardware/joint_command',
            qos_profile
        )

        # Set timer to publish at 100Hz (10ms)
        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('Joint Test Example started, waiting for first joint state...')

    def joint_state_callback(self, msg):
        """Callback when joint state is received"""
        if self.initial_positions is None:
            self.initial_positions = list(msg.position)
            self.get_logger().info(f'Received initial positions: {len(self.initial_positions)} joints')
            
            # Print initial positions
            self.print_positions("Initial positions", self.initial_positions)
            
            # Generate trajectories
            self.generate_trajectories()
            self.initialized = True
            self.get_logger().info('Trajectories generated, starting control loop')

    def generate_trajectories(self):
        """Generate linear interpolation trajectories for each joint"""
        if self.initial_positions is None:
            return

        num_joints = len(self.initial_positions)
        self.interpolated_positions = []

        for i in range(num_joints):
            if i < len(self.target_positions) and i < len(self.num_steps):
                start_pos = self.initial_positions[i]
                target_pos = self.target_positions[i]
                num_steps = int(self.num_steps[i])
                
                # Simple linear interpolation
                trajectory = []
                if num_steps > 1:
                    step_size = (target_pos - start_pos) / (num_steps - 1)
                    for j in range(num_steps):
                        trajectory.append(start_pos + step_size * j)
                else:
                    trajectory.append(target_pos)
                
                self.interpolated_positions.append(trajectory)
            else:
                # If no target specified, keep initial position
                self.interpolated_positions.append([self.initial_positions[i]])

    def print_positions(self, label, positions):
        """Print positions in a readable format"""
        pos_str = f"\n{label}:\n["
        for i, pos in enumerate(positions):
            pos_str += f"{pos:.3f}"
            if i < len(positions) - 1:
                pos_str += ", "
                # Add newline every 6 elements for better readability
                if (i + 1) % 6 == 0:
                    pos_str += "\n "
        pos_str += "]\n"
        self.get_logger().info(pos_str)

    def timer_callback(self):
        """Timer callback to publish joint commands at 100Hz"""
        if not self.initialized or self.initial_positions is None:
            return

        # Create joint command message
        msg = JointCommand()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'joint_test'

        num_joints = len(self.initial_positions)
        msg.position = [0.0] * num_joints
        msg.velocity = [0.0] * num_joints
        msg.feed_forward_torque = [0.0] * num_joints
        msg.torque = [0.0] * num_joints
        msg.stiffness = [self.stiffness] * num_joints
        msg.damping = [self.damping] * num_joints
        msg.parallel_parser_type = 0

        all_reached = True

        for i in range(num_joints):
            if not self.reached_targets[i]:
                if i < len(self.interpolated_positions) and i < len(self.current_steps):
                    if self.current_steps[i] < len(self.interpolated_positions[i]):
                        # Use interpolated position
                        msg.position[i] = self.interpolated_positions[i][self.current_steps[i]]
                        msg.velocity[i] = 0.0
                        self.current_steps[i] += 1
                    else:
                        # Hold at target position
                        if i < len(self.target_positions):
                            msg.position[i] = self.target_positions[i]
                        else:
                            msg.position[i] = self.initial_positions[i]
                        msg.velocity[i] = 0.0
                        self.reached_targets[i] = True
                else:
                    # Keep initial position if no trajectory
                    msg.position[i] = self.initial_positions[i]
                    msg.velocity[i] = 0.0
                    self.reached_targets[i] = True
                all_reached = False
            else:
                # Keep sending the target position
                if i < len(self.target_positions):
                    msg.position[i] = self.target_positions[i]
                else:
                    msg.position[i] = self.initial_positions[i]
                msg.velocity[i] = 0.0

        if all_reached:
            if not self.all_reached_logged:
                self.get_logger().info('All joints have reached their target positions!')
                self.all_reached_logged = True

        # Always publish joint command to maintain position
        self.joint_cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    joint_test_example = JointTestExample()

    try:
        rclpy.spin(joint_test_example)
    except KeyboardInterrupt:
        joint_test_example.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        # Destroy the node explicitly
        joint_test_example.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
