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
  bool UseSimplifiedGeometry() const { return use_simplified_geometry_; }
  std::string GetCollisionModelCondition() const;
  std::string GetXmlFilenameByCollisionType() const;  // 根据碰撞类型获取对应的XML文件名

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
  bool use_simplified_geometry_ = false;
  std::string simplified_xml_filename_;  // 简化几何体XML文件名
  std::string mesh_xml_filename_;        // 真实mesh XML文件名
};

#endif  // CONFIG_LOADER_H_
