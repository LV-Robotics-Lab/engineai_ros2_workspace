#!/usr/bin/env python3
"""
验证 MNN 模型配置和 YAML 参数的一致性
"""

import yaml
import os
import sys

def verify_config(yaml_path, mnn_path=None):
    """验证 YAML 配置和 MNN 模型的一致性"""
    print(f"验证配置文件: {yaml_path}")
    print("=" * 60)
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查基本参数
    num_observations = config.get('num_observations', 0)
    num_include_obs_steps = config.get('num_include_obs_steps', 0)
    num_clock_signal = config.get('num_clock_signal', 0)
    num_commands = config.get('num_commands', 0)
    active_joint_names = config.get('active_joint_names', [])
    num_actions = len(active_joint_names)
    
    # 计算预期维度
    expected_input_dim = num_observations * num_include_obs_steps + num_clock_signal + num_commands
    expected_output_dim = num_actions
    
    print(f"\n基本参数:")
    print(f"  num_observations: {num_observations}")
    print(f"  num_include_obs_steps: {num_include_obs_steps}")
    print(f"  num_clock_signal: {num_clock_signal}")
    print(f"  num_commands: {num_commands}")
    print(f"  active_joint_names 数量: {num_actions}")
    
    print(f"\n预期 MNN 模型维度:")
    print(f"  输入维度: {expected_input_dim}")
    print(f"    = {num_observations} * {num_include_obs_steps} + {num_clock_signal} + {num_commands}")
    print(f"    = {num_observations * num_include_obs_steps} + {num_clock_signal} + {num_commands}")
    print(f"  输出维度: {expected_output_dim}")
    print(f"    = num_actions = {num_actions}")
    
    # 检查参数数组长度
    print(f"\n参数数组验证:")
    
    # 检查 joint_kp
    joint_kp = config.get('joint_kp', [])
    joint_kp_flat = []
    for group in joint_kp:
        joint_kp_flat.extend(group)
    print(f"  joint_kp 总长度: {len(joint_kp_flat)} (应该是 {num_actions})")
    if len(joint_kp_flat) != num_actions:
        print(f"    ⚠️  警告: joint_kp 长度不匹配!")
    else:
        print(f"    ✓ joint_kp 长度正确")
    
    # 检查 action_scale
    action_scale = config.get('action_scale', [])
    action_scale_flat = []
    for group in action_scale:
        action_scale_flat.extend(group)
    print(f"  action_scale 总长度: {len(action_scale_flat)} (应该是 {num_actions})")
    if len(action_scale_flat) != num_actions:
        print(f"    ⚠️  警告: action_scale 长度不匹配!")
    else:
        print(f"    ✓ action_scale 长度正确")
    
    # 检查关节参数映射
    print(f"\n关节参数映射 (前10个):")
    print(f"{'索引':<6} {'关节名称':<25} {'joint_kp':<10} {'action_scale':<12}")
    print("-" * 60)
    for i in range(min(10, num_actions)):
        joint_name = active_joint_names[i] if i < len(active_joint_names) else "N/A"
        kp = joint_kp_flat[i] if i < len(joint_kp_flat) else "N/A"
        scale = action_scale_flat[i] if i < len(action_scale_flat) else "N/A"
        print(f"{i:<6} {joint_name:<25} {kp:<10} {scale:<12}")
    
    # 检查 MNN 文件
    if mnn_path:
        policy_file = config.get('policy_file', '')
        full_mnn_path = os.path.join(os.path.dirname(yaml_path), policy_file)
        if os.path.exists(full_mnn_path):
            print(f"\n✓ MNN 模型文件存在: {full_mnn_path}")
        else:
            print(f"\n⚠️  MNN 模型文件不存在: {full_mnn_path}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 verify_mnn_config.py <yaml_path> [mnn_path]")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    mnn_path = sys.argv[2] if len(sys.argv) > 2 else None
    verify_config(yaml_path, mnn_path)

