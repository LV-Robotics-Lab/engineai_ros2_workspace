cmake_minimum_required(VERSION 3.8)
project(interface_example)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND ENGINEAI_ROBOTICS_THIRD_PARTY /opt/engineai_robotics_third_party)
list(APPEND CMAKE_PREFIX_PATH ${ENGINEAI_ROBOTICS_THIRD_PARTY})
# Add conda environment to find glog
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(interface_protocol REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
# Find glog for csv_loader (only needed for rl_basic_example_CHR)
# Config package (apt: libgoogle-glog-dev) or conda-forge glog
find_package(glog CONFIG QUIET)
if(NOT TARGET glog::glog)
  find_package(Glog QUIET)
endif()
if(NOT TARGET glog::glog AND GLOG_INCLUDE_DIRS AND GLOG_LIBRARIES)
  add_library(glog::glog UNKNOWN IMPORTED)
  set_target_properties(glog::glog PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GLOG_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${GLOG_LIBRARIES}")
endif()
# Manual lookup: conda + Debian multiarch (lib under /usr/lib/x86_64-linux-gnu)
if(NOT TARGET glog::glog)
  find_path(GLOG_INCLUDE_DIR
    NAMES glog/logging.h
    PATHS
      $ENV{CONDA_PREFIX}/include
      /usr/include
      /usr/local/include
      ${ENGINEAI_ROBOTICS_THIRD_PARTY}/include
  )
  find_library(GLOG_LIBRARY
    NAMES glog
    PATHS
      $ENV{CONDA_PREFIX}/lib
      /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
      /usr/lib/aarch64-linux-gnu
      /usr/lib/x86_64-linux-gnu
      /usr/lib
      /usr/local/lib
      ${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib
  )
  if(GLOG_INCLUDE_DIR AND GLOG_LIBRARY)
    message(STATUS "Found glog manually: ${GLOG_INCLUDE_DIR}, ${GLOG_LIBRARY}")
    add_library(glog::glog UNKNOWN IMPORTED)
    set_target_properties(glog::glog PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GLOG_INCLUDE_DIR}"
      IMPORTED_LOCATION "${GLOG_LIBRARY}"
    )
  endif()
endif()
if(NOT TARGET glog::glog)
  message(FATAL_ERROR
    "glog not found. Install one of:\n"
    "  sudo apt install -y libgoogle-glog-dev\n"
    "  conda install -y -c conda-forge glog")
endif()

include_directories(
  include 
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add executable with message_handler source
add_executable(hold_joint_status
  src/hold_joint_status.cc
  src/components/message_handler.cc
)
ament_target_dependencies(hold_joint_status
  rclcpp
  interface_protocol
)

# Add joint_test_example executable
add_executable(joint_test_example
  src/joint_test_example.cc
  src/components/message_handler.cc
)
ament_target_dependencies(joint_test_example
  rclcpp
  interface_protocol
)
target_link_libraries(joint_test_example
  yaml-cpp::yaml-cpp
  Eigen3::Eigen
)

# Search source file in src/parameter/ and src/components/ and src/math/
file(GLOB_RECURSE rl_basic_example_src_files
  src/parameter/*.cc
  src/components/*.cc
  src/math/*.cc
  src/math/*.cpp
)

# Add csv_loader.cc for rl_basic_example_CHR (but not LoadProfileTrajectory which needs RlDanceParam)
set(rl_dance_csv_loader_src
  src/rl_dance/src/csv_loader.cc
)

add_executable(rl_basic_example
  src/rl_basic_example.cc
  ${rl_basic_example_src_files}
)

target_include_directories(rl_basic_example
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    ${ENGINEAI_ROBOTICS_THIRD_PARTY}/include
)

target_link_libraries(rl_basic_example
  yaml-cpp::yaml-cpp
  Eigen3::Eigen
  ${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib/libMNN.so
)

ament_target_dependencies(rl_basic_example
  rclcpp
  interface_protocol
)

# Add rl_basic_example_XZL executable
add_executable(rl_basic_example_XZL
  src/rl_basic_example_XZL.cc
  ${rl_basic_example_src_files}
)

target_include_directories(rl_basic_example_XZL
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    ${ENGINEAI_ROBOTICS_THIRD_PARTY}/include
)

target_link_libraries(rl_basic_example_XZL
  yaml-cpp::yaml-cpp
  Eigen3::Eigen
  ${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib/libMNN.so
)

ament_target_dependencies(rl_basic_example_XZL
  rclcpp
  interface_protocol
  std_msgs
)

# Add rl_basic_example_CHR executable
# Note: We don't link rl_dance library (it requires basic_param), 
# but we can use csv_loader header-only functions
add_executable(rl_basic_example_CHR
  src/rl_basic_example_CHR.cc
  ${rl_basic_example_src_files}
  ${rl_dance_csv_loader_src}
)

target_include_directories(rl_basic_example_CHR
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/rl_dance/include>
    ${ENGINEAI_ROBOTICS_THIRD_PARTY}/include
)
# Add conda glog include if available
if(DEFINED ENV{CONDA_PREFIX})
  target_include_directories(rl_basic_example_CHR PUBLIC $ENV{CONDA_PREFIX}/include)
endif()
# Define glog macro to fix include issues
target_compile_definitions(rl_basic_example_CHR PRIVATE GLOG_NO_ABBREVIATED_SEVERITIES)

target_link_libraries(rl_basic_example_CHR
  yaml-cpp::yaml-cpp
  Eigen3::Eigen
  ${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib/libMNN.so
  glog::glog  # Required for csv_loader.cc
  # Note: We don't link rl_dance library (it requires basic_param)
  # csv_loader functions are used as header-only
)

# Set rpath to find glog from conda environment at runtime
if(DEFINED ENV{CONDA_PREFIX})
  set_target_properties(rl_basic_example_CHR PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib:${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib"
  )
endif()

ament_target_dependencies(rl_basic_example_CHR
  rclcpp
  interface_protocol
)


install(TARGETS
  hold_joint_status
  joint_test_example
  rl_basic_example
  rl_basic_example_XZL
  rl_basic_example_CHR
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# 安装 config 文件
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# 安装 Python 脚本
install(PROGRAMS
  scripts/body_velocity_control_example.py
  src/joint_state_converter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
