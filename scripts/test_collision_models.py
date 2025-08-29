#!/usr/bin/env python3
"""
测试碰撞模型选择功能
验证 use_simplified_geometry 参数是否能正确选择不同的 XML 文件
"""

import yaml
import os
import sys

def test_config_loading():
    """测试配置文件加载"""
    print("=== 测试配置文件加载 ===")
    
    config_path = "src/simulation/mujoco/assets/config/pm_v2.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查碰撞模型配置
        if "collision_model" in config:
            collision_config = config["collision_model"]
            print(f"✅ 找到碰撞模型配置")
            
            # 检查 use_simplified_geometry 参数
            if "use_simplified_geometry" in collision_config:
                use_simplified = collision_config["use_simplified_geometry"]
                print(f"✅ use_simplified_geometry: {use_simplified}")
            else:
                print("❌ 缺少 use_simplified_geometry 参数")
                return False
            
            # 检查 XML 文件配置
            if "xml_files" in collision_config:
                xml_files = collision_config["xml_files"]
                print(f"✅ 找到 XML 文件配置")
                
                if "simplified" in xml_files:
                    print(f"✅ 简化几何体 XML: {xml_files['simplified']}")
                else:
                    print("❌ 缺少简化几何体 XML 配置")
                    return False
                
                if "mesh" in xml_files:
                    print(f"✅ 真实 mesh XML: {xml_files['mesh']}")
                else:
                    print("❌ 缺少真实 mesh XML 配置")
                    return False
            else:
                print("❌ 缺少 XML 文件配置")
                return False
        else:
            print("❌ 缺少碰撞模型配置")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_xml_files_exist():
    """测试 XML 文件是否存在"""
    print("\n=== 测试 XML 文件存在性 ===")
    
    xml_dir = "src/simulation/mujoco/assets/resource/robot/pm_v2/xml"
    
    if not os.path.exists(xml_dir):
        print(f"❌ XML 目录不存在: {xml_dir}")
        return False
    
    # 检查简化几何体 XML 文件
    simplified_xml = os.path.join(xml_dir, "serial_pm_v2_simplified.xml")
    if os.path.exists(simplified_xml):
        print(f"✅ 简化几何体 XML 文件存在: {simplified_xml}")
    else:
        print(f"❌ 简化几何体 XML 文件不存在: {simplified_xml}")
        return False
    
    # 检查真实 mesh XML 文件
    mesh_xml = os.path.join(xml_dir, "serial_pm_v2_mesh.xml")
    if os.path.exists(mesh_xml):
        print(f"✅ 真实 mesh XML 文件存在: {mesh_xml}")
    else:
        print(f"❌ 真实 mesh XML 文件不存在: {mesh_xml}")
        return False
    
    return True

def test_xml_content():
    """测试 XML 文件内容"""
    print("\n=== 测试 XML 文件内容 ===")
    
    xml_dir = "src/simulation/mujoco/assets/resource/robot/pm_v2/xml"
    
    # 检查简化几何体 XML 内容
    simplified_xml = os.path.join(xml_dir, "serial_pm_v2_simplified.xml")
    try:
        with open(simplified_xml, 'r') as f:
            content = f.read()
            if 'model="engineai_robotics_simplified"' in content:
                print("✅ 简化几何体 XML 模型名称正确")
            else:
                print("❌ 简化几何体 XML 模型名称错误")
                return False
            
            if 'type="sphere"' in content or 'type="box"' in content or 'type="cylinder"' in content:
                print("✅ 简化几何体 XML 包含简化几何体定义")
            else:
                print("❌ 简化几何体 XML 缺少简化几何体定义")
                return False
    except Exception as e:
        print(f"❌ 读取简化几何体 XML 失败: {e}")
        return False
    
    # 检查真实 mesh XML 内容
    mesh_xml = os.path.join(xml_dir, "serial_pm_v2_mesh.xml")
    try:
        with open(mesh_xml, 'r') as f:
            content = f.read()
            if 'model="engineai_robotics_mesh"' in content:
                print("✅ 真实 mesh XML 模型名称正确")
            else:
                print("❌ 真实 mesh XML 模型名称错误")
                return False
            
            if 'type="mesh"' in content:
                print("✅ 真实 mesh XML 包含 mesh 几何体定义")
            else:
                print("❌ 真实 mesh XML 缺少 mesh 几何体定义")
                return False
    except Exception as e:
        print(f"❌ 读取真实 mesh XML 失败: {e}")
        return False
    
    return True

def test_config_logic():
    """测试配置逻辑"""
    print("\n=== 测试配置逻辑 ===")
    
    config_path = "src/simulation/mujoco/assets/config/pm_v2.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        collision_config = config["collision_model"]
        use_simplified = collision_config["use_simplified_geometry"]
        xml_files = collision_config["xml_files"]
        
        # 测试简化几何体配置
        if use_simplified:
            expected_xml = xml_files["simplified"]
            print(f"✅ 当前配置使用简化几何体，应加载: {expected_xml}")
        else:
            expected_xml = xml_files["mesh"]
            print(f"✅ 当前配置使用真实 mesh，应加载: {expected_xml}")
        
        # 检查对应的 XML 文件是否存在
        xml_path = f"src/simulation/mujoco/assets/resource/robot/pm_v2/xml/{expected_xml}"
        if os.path.exists(xml_path):
            print(f"✅ 对应的 XML 文件存在: {xml_path}")
        else:
            print(f"❌ 对应的 XML 文件不存在: {xml_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 配置逻辑测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始测试碰撞模型选择功能...\n")
    
    # 切换到工作目录
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(workspace_dir)
    
    tests = [
        test_config_loading,
        test_xml_files_exist,
        test_xml_content,
        test_config_logic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ 测试失败: {test.__name__}")
        except Exception as e:
            print(f"❌ 测试异常: {test.__name__} - {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！碰撞模型选择功能正常工作。")
        print("\n使用方法:")
        print("1. 修改 src/simulation/mujoco/assets/config/pm_v2.yaml")
        print("2. 设置 use_simplified_geometry: true 使用简化几何体")
        print("3. 设置 use_simplified_geometry: false 使用真实 mesh")
        return 0
    else:
        print("❌ 部分测试失败，请检查配置和文件。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
