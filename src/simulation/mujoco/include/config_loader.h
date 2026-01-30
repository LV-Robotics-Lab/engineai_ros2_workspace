#ifndef CONFIG_LOADER_H_
#define CONFIG_LOADER_H_

#include <yaml-cpp/yaml.h>
#include <map>
#include <string>
#include <vector>

class ConfigLoader {
 public:
  ConfigLoader(const std::string& config_file);
  ~ConfigLoader() = default;

  bool LoadConfig();

  // Getters for various configuration parameters
  std::string GetUrdfFilename() const { return urdf_filename_; }
  std::string GetXmlFilename() const { return xml_filename_; }
  int GetNumTotalJoints() const { return num_total_joints_; }
  int GetNumContacts() const { return num_contacts_; }
  int GetNumSingleContactDimensions() const { return num_single_contact_dimensions_; }
  int GetMaxPerturbations() const { return max_perturbations_; }

  // Topic names
  std::string GetImuTopic() const { return imu_topic_; }
  std::string GetJointStateTopic() const { return joint_state_topic_; }
  std::string GetJointCommandTopic() const { return joint_command_topic_; }

  // 接触点可视化配置
  double GetContactPositionOffset() const { return contact_position_offset_; }
  double GetContactMarkerSize() const { return contact_marker_size_; }
  double GetContactForceScale() const { return contact_force_scale_; }
  bool IsContactVisualizationEnabled() const { return contact_visualization_enabled_; }

  // 碰撞模型配置
  std::string GetCollisionModelType() const { return collision_model_type_; }
  std::string GetCollisionModelCondition() const;
  std::string GetXmlFilenameByCollisionType() const;  // 根据碰撞类型获取对应的XML文件名

  // 干扰力配置
  double GetDefaultForceMagnitude() const { return default_force_magnitude_; }
  double GetDefaultTorqueMagnitude() const { return default_torque_magnitude_; }
  double GetForceStep() const { return force_step_; }
  double GetTorqueStep() const { return torque_step_; }
  double GetForceDuration() const { return force_duration_; }
  const std::vector<double>& GetPerturbationForce(const std::string& direction) const;
  const std::vector<double>& GetPerturbationForce(const std::string& direction, double magnitude) const;
  const std::vector<double>& GetPerturbationTorque(const std::string& direction) const;
  const std::vector<double>& GetPerturbationTorque(const std::string& direction, double magnitude) const;

  // 自动采样配置
  bool GetAutoSampling() const { return auto_sampling_; }
  const std::string& GetAutoDirection() const { return auto_direction_; }
  double GetAutoDelay() const { return auto_delay_; }

  // 初始速度配置
  double GetInitialLinearVelocityX() const { return initial_linear_velocity_x_; }
  double GetInitialLinearVelocityY() const { return initial_linear_velocity_y_; }
  double GetInitialLinearVelocityZ() const { return initial_linear_velocity_z_; }
  double GetInitialAngularVelocityX() const { return initial_angular_velocity_x_; }
  double GetInitialAngularVelocityY() const { return initial_angular_velocity_y_; }
  double GetInitialAngularVelocityZ() const { return initial_angular_velocity_z_; }

  // 防护功能配置
  bool IsProtectionEnabled() const { return protection_enabled_; }
  double GetProtectionThickness() const { return protection_thickness_; }

  // Asset path related methods
  std::string GetModelFilePath() const;
  std::string GetResourceDir() const;
  void SetAssetsPath(const std::string& assets_path) { assets_path_ = assets_path; }

 private:
  std::string config_file_;
  std::string assets_path_;  // Path to assets directory

  // Model parameters
  std::string urdf_filename_;
  std::string xml_filename_;
  int num_total_joints_ = 0;
  int num_contacts_ = 0;
  int num_single_contact_dimensions_ = 0;
  int max_perturbations_ = 2;  // 默认最大干扰力数量为2

  // Topic names
  std::string imu_topic_;
  std::string joint_state_topic_;
  std::string joint_command_topic_;

  // 接触点可视化配置
  double contact_position_offset_ = 0.0;
  double contact_marker_size_ = 0.03;
  double contact_force_scale_ = 0.01;
  bool contact_visualization_enabled_ = true;

  // 碰撞模型配置
  std::string collision_model_type_ = "simplified";  // 碰撞模型类型："simplified", "mesh", "mjlab", 或 "default"
  std::string simplified_xml_filename_;  // 简化几何体XML文件名
  std::string mesh_xml_filename_;        // 真实mesh XML文件名
  std::string mjlab_xml_filename_;       // mjlab XML文件名
  std::string default_xml_filename_;      // 默认XML文件名

  // 干扰力配置
  double default_force_magnitude_ = 20.0;    // 默认推力大小
  double default_torque_magnitude_ = 5.0;    // 默认扭矩大小
  double force_step_ = 20.0;                 // 推力调整步长
  double torque_step_ = 5.0;                 // 扭矩调整步长
  double force_duration_ = 0.2;              // 推力持续时间

  // 自动采样配置
  bool auto_sampling_ = false;               // 自动采样模式
  std::string auto_direction_ = "forward";   // 自动采样方向
  double auto_delay_ = 1.0;                  // 自动采样延迟时间

  // 初始速度配置
  double initial_linear_velocity_x_ = 0.0;   // 初始线速度X分量
  double initial_linear_velocity_y_ = 0.0;   // 初始线速度Y分量
  double initial_linear_velocity_z_ = 0.0;   // 初始线速度Z分量
  double initial_angular_velocity_x_ = 0.0;  // 初始角速度X分量
  double initial_angular_velocity_y_ = 0.0;  // 初始角速度Y分量
  double initial_angular_velocity_z_ = 0.0;  // 初始角速度Z分量

  // 防护功能配置
  bool protection_enabled_ = false;          // 是否启用防护功能（虚拟护具）
  double protection_thickness_ = 12.0;       // 护具厚度（mm），默认12mm
};

#endif  // CONFIG_LOADER_H_
