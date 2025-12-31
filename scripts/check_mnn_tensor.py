#!/usr/bin/env python3
"""
检查 MNN 模型的输入输出 tensor 信息
"""

import sys
import os

# 添加 MNN Python 绑定路径（如果可用）
try:
    import MNN
    import numpy as np
    
    def check_mnn_model(model_path):
        """检查 MNN 模型的 tensor 信息"""
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return
        
        print(f"检查模型: {model_path}")
        print("=" * 60)
        
        # 创建解释器
        interpreter = MNN.Interpreter.createFromFile(model_path)
        
        # 创建会话
        config = MNN.ScheduleConfig()
        config.numThread = 1
        session = interpreter.createSession(config)
        
        # 获取输入 tensor
        input_tensor = interpreter.getSessionInput(session, None)
        if input_tensor:
            input_shape = input_tensor.getShape()
            print(f"输入 tensor:")
            print(f"  形状: {input_shape}")
            print(f"  维度: {input_shape[-1] if input_shape else 'N/A'}")
            print(f"  总元素数: {np.prod(input_shape) if input_shape else 'N/A'}")
        
        # 获取输出 tensor
        output_tensor = interpreter.getSessionOutput(session, None)
        if output_tensor:
            output_shape = output_tensor.getShape()
            print(f"\n输出 tensor:")
            print(f"  形状: {output_shape}")
            print(f"  维度: {output_shape[-1] if output_shape else 'N/A'}")
            print(f"  总元素数: {np.prod(output_shape) if output_shape else 'N/A'}")
        
        # 获取所有输入输出名称
        print(f"\n所有输入 tensor 名称:")
        input_names = interpreter.getSessionInputAll(session)
        for name, tensor in input_names.items():
            shape = tensor.getShape()
            print(f"  {name}: {shape}")
        
        print(f"\n所有输出 tensor 名称:")
        output_names = interpreter.getSessionOutputAll(session)
        for name, tensor in output_names.items():
            shape = tensor.getShape()
            print(f"  {name}: {shape}")
        
        interpreter.releaseSession(session)
        interpreter.releaseModel()
        
    if __name__ == "__main__":
        if len(sys.argv) < 2:
            print("用法: python3 check_mnn_tensor.py <model_path>")
            sys.exit(1)
        
        model_path = sys.argv[1]
        check_mnn_model(model_path)
        
except ImportError:
    print("警告: MNN Python 绑定不可用，无法直接检查模型")
    print("请使用 MNN 工具或 C++ 代码检查模型 tensor 信息")
    print("\n或者，可以通过代码中的 tensor shape 来推断：")
    print("  - 输入维度 = num_observations * num_include_obs_steps + num_clock_signal + num_commands")
    print("  - 输出维度 = num_actions (应该等于 active_joint_names 的数量)")

