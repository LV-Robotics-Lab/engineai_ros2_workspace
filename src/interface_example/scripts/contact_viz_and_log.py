#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interface_protocol.msg import ContactForce
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import csv
import os

def force_to_color(force, max_force=200.0):
    # 力越大，颜色越红，力小为蓝色
    norm = min(abs(force) / max_force, 1.0)
    return ColorRGBA(r=norm, g=0.0, b=1.0-norm, a=1.0)

class ContactVisualizer(Node):
    def __init__(self):
        super().__init__('contact_visualizer')
        self.subscription = self.create_subscription(
            ContactForce,
            '/mujoco/contact_forces',
            self.listener_callback,
            10)
        self.marker_pub = self.create_publisher(MarkerArray, '/contact_markers', 10)
        self.csv_file = os.path.expanduser('~/contact_points_log.csv')
        self.csv_header_written = False

    def listener_callback(self, msg):
        marker_array = MarkerArray()
        with open(self.csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not self.csv_header_written:
                header = [
                    'stamp', 'contact_name', 'x', 'y', 'z',
                    'fx', 'fy', 'fz', 'gap', 'body1', 'body2'
                ]
                writer.writerow(header)
                self.csv_header_written = True

            for i, name in enumerate(msg.contact_names):
                x = msg.contact_positions_x[i]
                y = msg.contact_positions_y[i]
                z = msg.contact_positions_z[i]
                fx = msg.contact_forces_x[i]
                fy = msg.contact_forces_y[i]
                fz = msg.contact_forces_z[i]
                gap = msg.contact_gaps[i]
                body1 = msg.contact_bodies_1[i]
                body2 = msg.contact_bodies_2[i]
                force_mag = (fx**2 + fy**2 + fz**2) ** 0.5

                # 写入csv
                writer.writerow([
                    msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    name, x, y, z, fx, fy, fz, gap, body1, body2
                ])

                # RViz Marker
                marker = Marker()
                marker.header = msg.header
                marker.ns = "contact_points"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                marker.color = force_to_color(force_mag)
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = int(1e8)  # 0.1秒
                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ContactVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()