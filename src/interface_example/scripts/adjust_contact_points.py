#!/usr/bin/env python3
"""
接触点显示参数调整工具

这个脚本帮助用户调整MuJoCo中接触点的显示位置，使其更准确地显示在机器人mesh表面。
"""

import yaml
import os
import sys
import argparse
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"错误：无法加载配置文件 {config_path}: {e}")
        return None

def save_config(config, config_path):
    """保存配置文件"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"配置文件已保存到: {config_path}")
        return True
    except Exception as e:
        print(f"错误：无法保存配置文件 {config_path}: {e}")
        return False

def adjust_contact_position_offset(config_path, offset):
    """调整接触点位置偏移量"""
    config = load_config(config_path)
    if config is None:
        return False
    
    # 确保contact_visualization节点存在
    if 'contact_visualization' not in config:
        config['contact_visualization'] = {}
    
    # 设置位置偏移量
    config['contact_visualization']['position_offset'] = offset
    
    print(f"接触点位置偏移量已设置为: {offset} 米")
    print("说明:")
    print("  - 正值：接触点向法向量方向偏移（远离表面）")
    print("  - 负值：接触点向法向量反方向偏移（靠近表面）")
    print("  - 0.0：使用MuJoCo原始计算的接触点位置")
    
    return save_config(config, config_path)

def adjust_contact_marker_size(config_path, size):
    """调整接触点标记大小"""
    config = load_config(config_path)
    if config is None:
        return False
    
    # 确保contact_visualization节点存在
    if 'contact_visualization' not in config:
        config['contact_visualization'] = {}
    
    # 设置标记大小
    config['contact_visualization']['marker_size'] = size
    
    print(f"接触点标记大小已设置为: {size} 米")
    
    return save_config(config, config_path)

def show_current_settings(config_path):
    """显示当前设置"""
    config = load_config(config_path)
    if config is None:
        return False
    
    print("当前接触点可视化设置:")
    print("-" * 40)
    
    if 'contact_visualization' in config:
        cv = config['contact_visualization']
        print(f"位置偏移量: {cv.get('position_offset', 0.0)} 米")
        print(f"标记大小: {cv.get('marker_size', 0.03)} 米")
        print(f"力显示比例: {cv.get('force_scale', 0.01)}")
        print(f"是否启用: {cv.get('enabled', True)}")
    else:
        print("未找到接触点可视化配置，使用默认值")
        print("位置偏移量: 0.0 米")
        print("标记大小: 0.03 米")
        print("力显示比例: 0.01")
        print("是否启用: True")
    
    return True

def interactive_adjustment(config_path):
    """交互式调整"""
    print("交互式接触点参数调整")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 显示当前设置")
        print("2. 调整位置偏移量")
        print("3. 调整标记大小")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            show_current_settings(config_path)
        elif choice == '2':
            try:
                offset = float(input("请输入位置偏移量（米，建议范围 -0.01 到 0.01）: "))
                adjust_contact_position_offset(config_path, offset)
            except ValueError:
                print("错误：请输入有效的数字")
        elif choice == '3':
            try:
                size = float(input("请输入标记大小（米，建议范围 0.01 到 0.1）: "))
                adjust_contact_marker_size(config_path, size)
            except ValueError:
                print("错误：请输入有效的数字")
        elif choice == '4':
            print("退出调整工具")
            break
        else:
            print("无效选择，请重新输入")

def main():
    parser = argparse.ArgumentParser(description='MuJoCo接触点显示参数调整工具')
    parser.add_argument('--config', type=str, 
                       default='/opt/ros/humble/share/mujoco_simulator/assets/config/pm_v2.yaml',
                       help='配置文件路径')
    parser.add_argument('--offset', type=float, help='设置位置偏移量（米）')
    parser.add_argument('--size', type=float, help='设置标记大小（米）')
    parser.add_argument('--show', action='store_true', help='显示当前设置')
    parser.add_argument('--interactive', action='store_true', help='交互式调整')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        print("请确保MuJoCo仿真器已正确安装")
        return 1
    
    if args.show:
        show_current_settings(args.config)
    elif args.offset is not None:
        adjust_contact_position_offset(args.config, args.offset)
    elif args.size is not None:
        adjust_contact_marker_size(args.config, args.size)
    elif args.interactive:
        interactive_adjustment(args.config)
    else:
        print("请指定操作。使用 --help 查看帮助信息")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 