#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from interface_protocol.msg import JointState as InterfaceJointState
from sensor_msgs.msg import JointState as SensorJointState
from std_msgs.msg import Header


class JointStateConverter(Node):
    def __init__(self):
        super().__init__('joint_state_converter')
        
        # 创建订阅者，订阅interface_protocol的JointState
        self.subscription = self.create_subscription(
            InterfaceJointState,
            '/hardware/joint_state',
            self.joint_state_callback,
            10
        )
        
        # 创建发布者，发布sensor_msgs的JointState
        self.publisher = self.create_publisher(
            SensorJointState,
            '/joint_states',
            10
        )
        
        self.get_logger().info('JointState转换节点已启动')
    
    def joint_state_callback(self, msg):
        # 创建sensor_msgs/JointState消息
        sensor_msg = SensorJointState()
        
        # 设置时间戳
        sensor_msg.header = Header()
        sensor_msg.header.stamp = self.get_clock().now().to_msg()
        sensor_msg.header.frame_id = 'base_link'
        
        # 复制关节名称
        sensor_msg.name = msg.name
        
        # 复制关节位置
        sensor_msg.position = msg.position
        
        # 复制关节速度
        sensor_msg.velocity = msg.velocity
        
        # 复制关节力矩
        sensor_msg.effort = msg.torque
        
        # 发布转换后的消息
        self.publisher.publish(sensor_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main() 