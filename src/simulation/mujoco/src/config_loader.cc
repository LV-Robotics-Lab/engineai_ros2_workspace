/**
 * @file config_loader.cc
 * @brief 配置文件加载器实现
 * @details 该文件实现了YAML配置文件的加载和解析功能，包括：
 *          - 模型参数加载（URDF、XML文件路径）
 *          - 传感器和执行器话题配置
 *          - 接触力可视化参数
 *          - 碰撞模型配置
 *          - 资源路径管理
 */

#include "config_loader.h"
#include <fstream>
#include <iostream>
#include <string>
#include <exception>

/**
 * @brief 构造函数
 * @param config_file 配置文件路径
 * @details 初始化配置加载器，设置配置文件路径
 */
ConfigLoader::ConfigLoader(const std::string& config_file) : config_file_(config_file) {
  LoadConfig();
}

/**
 * @brief 加载配置文件
 * @return 加载是否成功
 * @details 从YAML配置文件中加载所有必要的参数，包括：
 *          - 模型文件路径（URDF、XML）
 *          - 模型参数（关节数量、接触点数量等）
 *          - ROS话题名称
 *          - 接触力可视化参数
 *          - 碰撞模型配置
 */
bool ConfigLoader::LoadConfig() {
  try {
    // 加载YAML文件
    YAML::Node config = YAML::LoadFile(config_file_);

    // 加载执行器配置
    if (config["actuator"]) {
      YAML::Node actuator = config["actuator"];
      if (actuator["joint_command_topic"]) {
        joint_command_topic_ = actuator["joint_command_topic"].as<std::string>();
      }
      if (actuator["joint_state_topic"]) {
        joint_state_topic_ = actuator["joint_state_topic"].as<std::string>();
      }
    }

    // 加载碰撞模型配置
    if (config["collision_model"]) {
      YAML::Node collision_model = config["collision_model"];
      
      // 是否使用简化几何体进行碰撞检测
      if (collision_model["use_simplified_geometry"]) {
        use_simplified_geometry_ = collision_model["use_simplified_geometry"].as<bool>();
      }
      
      // 加载XML文件配置
      if (collision_model["xml_files"]) {
        YAML::Node xml_files = collision_model["xml_files"];
        if (xml_files["simplified"]) {
          simplified_xml_filename_ = xml_files["simplified"].as<std::string>();
        }
        if (xml_files["mesh"]) {
          mesh_xml_filename_ = xml_files["mesh"].as<std::string>();
        }
      }
    }

    // 加载接触力可视化配置
    if (config["contact_visualization"]) {
      YAML::Node contact_viz = config["contact_visualization"];
      if (contact_viz["enabled"]) {
        contact_visualization_enabled_ = contact_viz["enabled"].as<bool>();
      }
      if (contact_viz["force_scale"]) {
        contact_force_scale_ = contact_viz["force_scale"].as<double>();
      }
      if (contact_viz["marker_size"]) {
        contact_marker_size_ = contact_viz["marker_size"].as<double>();
      }
      if (contact_viz["position_offset"]) {
        contact_position_offset_ = contact_viz["position_offset"].as<double>();
      }
    }

    // 加载模型参数
    if (config["model_param"]) {
      YAML::Node model_param = config["model_param"];
      if (model_param["num_contacts"]) {
        num_contacts_ = model_param["num_contacts"].as<int>();
      }
      if (model_param["num_single_contact_dimensions"]) {
        num_single_contact_dimensions_ = model_param["num_single_contact_dimensions"].as<int>();
      }
      if (model_param["num_total_joints"]) {
        num_total_joints_ = model_param["num_total_joints"].as<int>();
      }
      if (model_param["max_perturbations"]) {
        max_perturbations_ = model_param["max_perturbations"].as<int>();
      }
    }

    // 加载传感器配置
    if (config["sensor"]) {
      YAML::Node sensor = config["sensor"];
      if (sensor["imu_topic"]) {
        imu_topic_ = sensor["imu_topic"].as<std::string>();
      }
    }

    // 加载干扰力配置
    if (config["perturbation"]) {
      YAML::Node perturbation = config["perturbation"];
      if (perturbation["default_force_magnitude"]) {
        default_force_magnitude_ = perturbation["default_force_magnitude"].as<double>();
      }
      if (perturbation["default_torque_magnitude"]) {
        default_torque_magnitude_ = perturbation["default_torque_magnitude"].as<double>();
      }
      if (perturbation["force_step"]) {
        force_step_ = perturbation["force_step"].as<double>();
      }
      if (perturbation["torque_step"]) {
        torque_step_ = perturbation["torque_step"].as<double>();
      }
      if (perturbation["force_duration"]) {
        force_duration_ = perturbation["force_duration"].as<double>();
      }
      // 读取自动采样配置
      if (perturbation["auto_sampling"]) {
        auto_sampling_ = perturbation["auto_sampling"].as<bool>();
      }
      if (perturbation["auto_direction"]) {
        auto_direction_ = perturbation["auto_direction"].as<std::string>();
      }
      if (perturbation["auto_delay"]) {
        auto_delay_ = perturbation["auto_delay"].as<double>();
      }
    }

    // 解析初始速度配置
    if (config["initial_velocity"]) {
      YAML::Node initial_velocity = config["initial_velocity"];
      if (initial_velocity["linear_x"]) {
        initial_linear_velocity_x_ = initial_velocity["linear_x"].as<double>();
      }
      if (initial_velocity["linear_y"]) {
        initial_linear_velocity_y_ = initial_velocity["linear_y"].as<double>();
      }
      if (initial_velocity["linear_z"]) {
        initial_linear_velocity_z_ = initial_velocity["linear_z"].as<double>();
      }
      if (initial_velocity["angular_x"]) {
        initial_angular_velocity_x_ = initial_velocity["angular_x"].as<double>();
      }
      if (initial_velocity["angular_y"]) {
        initial_angular_velocity_y_ = initial_velocity["angular_y"].as<double>();
      }
      if (initial_velocity["angular_z"]) {
        initial_angular_velocity_z_ = initial_velocity["angular_z"].as<double>();
      }
    }

    // 加载模型文件配置
    if (config["urdf"]) {
      urdf_filename_ = config["urdf"].as<std::string>();
    }
    if (config["xml"]) {
      xml_filename_ = config["xml"].as<std::string>();
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error loading config file: " << e.what() << std::endl;
    return false;
  }
}

/**
 * @brief 获取模型文件完整路径
 * @return 模型文件的完整路径
 * @details 返回assets_path_ + "/resource/" + xml_filename_的组合路径
 */
std::string ConfigLoader::GetModelFilePath() const { 
  return assets_path_ + "/resource/" + xml_filename_; 
}

/**
 * @brief 获取资源目录路径
 * @return 资源目录的完整路径
 * @details 返回assets_path_ + "/resource"的组合路径
 */
std::string ConfigLoader::GetResourceDir() const { 
  return assets_path_ + "/resource"; 
}

/**
 * @brief 获取碰撞模型条件字符串
 * @return 碰撞模型类型字符串
 * @details 根据use_simplified_geometry_的值返回"simplified"或"mesh"
 *          - simplified: 使用简化几何体进行碰撞检测
 *          - mesh: 使用原始网格进行碰撞检测
 */
std::string ConfigLoader::GetCollisionModelCondition() const {
  return use_simplified_geometry_ ? "simplified" : "mesh";
}

/**
 * @brief 根据碰撞类型获取对应的XML文件名
 * @return XML文件名
 * @details 根据use_simplified_geometry_的值返回对应的XML文件名
 *          - true: 返回简化几何体XML文件名
 *          - false: 返回真实mesh XML文件名
 */
std::string ConfigLoader::GetXmlFilenameByCollisionType() const {
  if (use_simplified_geometry_) {
    return "pm_v2_simplified.xml";  // 简化几何体版本的主配置文件
  } else {
    return "pm_v2_mesh.xml";        // Mesh版本的主配置文件
  }
}

/**
 * @brief 获取指定方向的推力配置
 * @param direction 推力方向（forward, backward, left, right, up, down）
 * @return 推力向量（3D向量）
 * @details 根据方向自动分配推力，使用默认推力大小
 */
const std::vector<double>& ConfigLoader::GetPerturbationForce(const std::string& direction) const {
  // 使用静态变量缓存各方向的推力向量
  static std::map<std::string, std::vector<double>> cached_forces;
  
  // 如果缓存中没有该方向，则创建
  if (cached_forces.find(direction) == cached_forces.end()) {
    std::vector<double> force(3, 0.0);
    
    if (direction == "forward") {
      force[0] = default_force_magnitude_;   // X轴正方向
    } else if (direction == "backward") {
      force[0] = -default_force_magnitude_;  // X轴负方向
    } else if (direction == "left") {
      force[1] = default_force_magnitude_;   // Y轴正方向
    } else if (direction == "right") {
      force[1] = -default_force_magnitude_;  // Y轴负方向
    } else if (direction == "up") {
      force[2] = default_force_magnitude_;   // Z轴正方向
    } else if (direction == "down") {
      force[2] = -default_force_magnitude_;  // Z轴负方向
    } else {
      // 未知方向，返回前向推力
      force[0] = default_force_magnitude_;
    }
    
    cached_forces[direction] = force;
  }
  
  return cached_forces[direction];
}

/**
 * @brief 获取指定方向的推力配置（使用指定大小）
 * @param direction 推力方向（forward, backward, left, right, up, down）
 * @param magnitude 推力大小
 * @return 推力向量（3D向量）
 * @details 根据方向自动分配推力，使用指定的推力大小
 */
const std::vector<double>& ConfigLoader::GetPerturbationForce(const std::string& direction, double magnitude) const {
  // 使用静态变量缓存各方向的推力向量（带大小参数）
  static std::map<std::string, std::vector<double>> cached_forces_with_magnitude;
  
  // 创建带大小信息的键
  std::string key = direction + "_" + std::to_string(magnitude);
  
  // 如果缓存中没有该键，则创建
  if (cached_forces_with_magnitude.find(key) == cached_forces_with_magnitude.end()) {
    std::vector<double> force(3, 0.0);
    
    if (direction == "forward") {
      force[0] = magnitude;   // X轴正方向
    } else if (direction == "backward") {
      force[0] = -magnitude;  // X轴负方向
    } else if (direction == "left") {
      force[1] = magnitude;   // Y轴正方向
    } else if (direction == "right") {
      force[1] = -magnitude;  // Y轴负方向
    } else if (direction == "up") {
      force[2] = magnitude;   // Z轴正方向
    } else if (direction == "down") {
      force[2] = -magnitude;  // Z轴负方向
    } else {
      // 未知方向，返回前向推力
      force[0] = magnitude;
    }
    
    cached_forces_with_magnitude[key] = force;
  }
  
  return cached_forces_with_magnitude[key];
}

/**
 * @brief 获取指定方向的扭矩配置
 * @param direction 扭矩方向（x_positive, x_negative, y_positive, y_negative, z_positive, z_negative）
 * @return 扭矩向量（3D向量）
 * @details 根据方向自动分配扭矩，使用默认扭矩大小
 */
const std::vector<double>& ConfigLoader::GetPerturbationTorque(const std::string& direction) const {
  // 使用静态变量缓存各方向的扭矩向量
  static std::map<std::string, std::vector<double>> cached_torques;
  
  // 如果缓存中没有该方向，则创建
  if (cached_torques.find(direction) == cached_torques.end()) {
    std::vector<double> torque(3, 0.0);
    
    if (direction == "x_positive") {
      torque[0] = default_torque_magnitude_;   // X轴正方向
    } else if (direction == "x_negative") {
      torque[0] = -default_torque_magnitude_;  // X轴负方向
    } else if (direction == "y_positive") {
      torque[1] = default_torque_magnitude_;   // Y轴正方向
    } else if (direction == "y_negative") {
      torque[1] = -default_torque_magnitude_;  // Y轴负方向
    } else if (direction == "z_positive") {
      torque[2] = default_torque_magnitude_;   // Z轴正方向
    } else if (direction == "z_negative") {
      torque[2] = -default_torque_magnitude_;  // Z轴负方向
    } else {
      // 未知方向，返回X轴正方向扭矩
      torque[0] = default_torque_magnitude_;
    }
    
    cached_torques[direction] = torque;
  }
  
  return cached_torques[direction];
}

/**
 * @brief 获取指定方向的扭矩配置（使用指定大小）
 * @param direction 扭矩方向（x_positive, x_negative, y_positive, y_negative, z_positive, z_negative）
 * @param magnitude 扭矩大小
 * @return 扭矩向量（3D向量）
 * @details 根据方向自动分配扭矩，使用指定的扭矩大小
 */
const std::vector<double>& ConfigLoader::GetPerturbationTorque(const std::string& direction, double magnitude) const {
  // 使用静态变量缓存各方向的扭矩向量（带大小参数）
  static std::map<std::string, std::vector<double>> cached_torques_with_magnitude;
  
  // 创建带大小信息的键
  std::string key = direction + "_" + std::to_string(magnitude);
  
  // 如果缓存中没有该键，则创建
  if (cached_torques_with_magnitude.find(key) == cached_torques_with_magnitude.end()) {
    std::vector<double> torque(3, 0.0);
    
    if (direction == "x_positive") {
      torque[0] = magnitude;   // X轴正方向
    } else if (direction == "x_negative") {
      torque[0] = -magnitude;  // X轴负方向
    } else if (direction == "y_positive") {
      torque[1] = magnitude;   // Y轴正方向
    } else if (direction == "y_negative") {
      torque[1] = -magnitude;  // Y轴负方向
    } else if (direction == "z_positive") {
      torque[2] = magnitude;   // Z轴正方向
    } else if (direction == "z_negative") {
      torque[2] = -magnitude;  // Z轴负方向
    } else {
      // 未知方向，返回X轴正方向扭矩
      torque[0] = magnitude;
    }
    
    cached_torques_with_magnitude[key] = torque;
  }
  
  return cached_torques_with_magnitude[key];
}

