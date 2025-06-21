#include <rclcpp/rclcpp.hpp>
#include <cstring>
#include "sim_manager.h"

int main(int argc, char** argv) {
  // 解析命令行参数
  bool export_contact = false;
  std::string contact_topic = "/mujoco/contact_forces";
  bool save_contact_csv = false;
  std::string csv_file_path = "";
  
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--export_contact") == 0) {
      export_contact = true;
    } else if (strcmp(argv[i], "--contact_topic") == 0 && i + 1 < argc) {
      contact_topic = argv[++i];
    } else if (strcmp(argv[i], "--save_contact_csv") == 0) {
      save_contact_csv = true;
    } else if (strcmp(argv[i], "--csv_file_path") == 0 && i + 1 < argc) {
      csv_file_path = argv[++i];
    }
  }

  rclcpp::init(argc, argv);
  auto& sim_manager = SimManager::GetInstance();

  if (!sim_manager.Initialize()) {
    return 1;
  }

  sim_manager.Run();
  rclcpp::shutdown();
  return 0;
}
