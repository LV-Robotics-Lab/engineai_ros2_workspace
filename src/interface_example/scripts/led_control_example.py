#!/usr/bin/env python3
"""
LED Control Example - Switch different LED light modes every 1 second
"""

import rclpy
from rclpy.node import Node
from interface_protocol.msg import LedControl


class LedControlExample(Node):
    """LED Control Example Node"""

    def __init__(self):
        super().__init__("led_control_example")

        # Create LED control publisher
        self.led_pub = self.create_publisher(
            LedControl,
            "/hardware/led_control",
            10,  # QoS profile depth
        )

        # Define all available LED modes (in order)
        self.led_modes = [
            LedControl.BLINK_RED,
            LedControl.BLINK_GREEN,
            LedControl.BLINK_BLUE,
            LedControl.BLINK_WHITE,
            LedControl.CONSTANT_ON_WHITE,
            LedControl.CONSTANT_ON_GREEN,
            LedControl.BREATHE_WHITE,
            LedControl.WATER_WHITE,
            LedControl.BREATHE_RED,
            LedControl.BLINK_ORANGE,
            LedControl.CONSTANT_ON_ORANGE,
        ]

        # Current LED mode index
        self.current_mode_index = 0

        # Mode name mapping (for log output)
        self.mode_names = {
            LedControl.BLINK_RED: "Blink Red",
            LedControl.BLINK_GREEN: "Blink Green",
            LedControl.BLINK_BLUE: "Blink Blue",
            LedControl.BLINK_WHITE: "Blink White",
            LedControl.CONSTANT_ON_WHITE: "Constant On White",
            LedControl.CONSTANT_ON_GREEN: "Constant On Green",
            LedControl.BREATHE_WHITE: "Breathe White",
            LedControl.WATER_WHITE: "Water White",
            LedControl.BREATHE_RED: "Breathe Red",
            LedControl.BLINK_ORANGE: "Blink Orange",
            LedControl.CONSTANT_ON_ORANGE: "Constant On Orange",
        }

        # Create timer to switch every 1 second
        self.timer_period = 1.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("LED control example node started")
        self.get_logger().info(
            f"Will switch LED mode every {self.timer_period} seconds"
        )

    def timer_callback(self):
        """Timer callback function - publish LED control message"""
        # Create LED control message
        msg = LedControl()
        msg.color = self.led_modes[self.current_mode_index]

        # Publish message
        self.led_pub.publish(msg)

        # Get current mode name
        mode_name = self.mode_names.get(msg.color, f"Unknown mode (0x{msg.color:x})")

        self.get_logger().info(
            f"Publish LED control: {mode_name} (mode value: 0x{msg.color:x})"
        )

        # Switch to next mode (cycle)
        self.current_mode_index = (self.current_mode_index + 1) % len(self.led_modes)


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    node = LedControlExample()

    try:
        rclpy.spin(node)
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
    exit(main())
