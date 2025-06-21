#!/usr/bin/env python3
"""
测试接触点CSV保存功能的脚本
"""

import rclpy
from rclpy.node import Node
from interface_protocol.msg import ContactForce
import csv
import os
from datetime import datetime

class ContactCSVTester(Node):
    def __init__(self):
        super().__init__('contact_csv_tester')
        
        # 创建CSV文件
        self.csv_filename = f"logs/contact_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('logs', exist_ok=True)
        
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 写入CSV头部
        self.csv_writer.writerow([
            'timestamp', 'contact_id', 'geom1_name', 'geom2_name',
            'pos_x', 'pos_y', 'pos_z', 'force_x', 'force_y', 'force_z',
            'torque_x', 'torque_y', 'torque_z', 'gap', 'body1_id', 'body2_id'
        ])
        
        # 订阅接触力话题
        self.subscription = self.create_subscription(
            ContactForce,
            '/mujoco/contact_forces',
            self.contact_callback,
            10
        )
        
        self.get_logger().info(f'Contact CSV tester started. Saving to: {self.csv_filename}')
        self.contact_count = 0
    
    def contact_callback(self, msg):
        """处理接触力消息并保存到CSV"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        for i in range(len(msg.contact_names)):
            self.csv_writer.writerow([
                timestamp,
                i,
                msg.contact_names[i],
                f"contact_{i}",
                msg.contact_positions_x[i],
                msg.contact_positions_y[i],
                msg.contact_positions_z[i],
                msg.contact_forces_x[i],
                msg.contact_forces_y[i],
                msg.contact_forces_z[i],
                msg.contact_torques_x[i],
                msg.contact_torques_y[i],
                msg.contact_torques_z[i],
                msg.contact_gaps[i],
                msg.contact_bodies_1[i],
                msg.contact_bodies_2[i]
            ])
        
        self.contact_count += 1
        if self.contact_count % 100 == 0:
            self.get_logger().info(f'Processed {self.contact_count} contact messages')
            self.csv_file.flush()
    
    def __del__(self):
        """析构函数，确保文件正确关闭"""
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
            self.get_logger().info(f'CSV file closed: {self.csv_filename}')

def main(args=None):
    rclpy.init(args=args)
    
    tester = ContactCSVTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 