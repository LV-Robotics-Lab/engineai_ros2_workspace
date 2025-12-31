#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class ZeroJointStatePublisher(Node):
    def __init__(self):
        super().__init__('zero_joint_state_publisher')
        
        # 发布者
        self.publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # 所有可动关节名称（从URDF中提取的revolute关节）
        self.joint_names = [
            'j00_hip_pitch_l', 'j01_hip_roll_l', 'j02_hip_yaw_l',
            'j03_knee_pitch_l', 'j04_ankle_pitch_l', 'j05_ankle_roll_l',
            'j06_hip_pitch_r', 'j07_hip_roll_r', 'j08_hip_yaw_r',
            'j09_knee_pitch_r', 'j10_ankle_pitch_r', 'j11_ankle_roll_r',
            'j13_shoulder_pitch_l', 'j14_shoulder_roll_l', 'j15_shoulder_yaw_l',
            'j16_elbow_pitch_l', 'j17_elbow_yaw_l',
            'j18_shoulder_pitch_r', 'j19_shoulder_roll_r', 'j20_shoulder_yaw_r',
            'j21_elbow_pitch_r', 'j22_elbow_yaw_r'
        ]
        
        # 创建定时器，以50Hz频率发布
        self.timer = self.create_timer(0.02, self.publish_joint_states)
        
        self.get_logger().info('零位关节状态发布节点已启动')
    
    def publish_joint_states(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'link_base'
        
        msg.name = self.joint_names
        msg.position = [0.0] * len(self.joint_names)  # 所有关节位置为0
        msg.velocity = [0.0] * len(self.joint_names)  # 所有关节速度为0
        msg.effort = [0.0] * len(self.joint_names)    # 所有关节力矩为0
        
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ZeroJointStatePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
        rclpy.shutdown()
        except Exception:
            # 如果已经 shutdown，忽略错误
            pass


if __name__ == '__main__':
    main()

