# 如何在CMakeLists.txt中添加cnpy依赖

## 方法1: 系统已安装cnpy

如果你的系统已经安装了cnpy库，在 `src/runner/rl_dance/CMakeLists.txt` 中添加：

```cmake
# 查找cnpy包
find_package(cnpy REQUIRED)

# 在target_link_libraries中添加cnpy
target_link_libraries(rl_dance_runner
    PUBLIC
        Eigen3::Eigen
        glog::glog
        cnpy  # 添加这一行
        # ... 其他依赖
)
```

## 方法2: 作为第三方库包含

如果系统没有安装cnpy，可以将其作为第三方库：

### 步骤1: 下载cnpy到third_party目录

```bash
cd /home/lby/engineai_lby/repo/engineai_robotics
mkdir -p engineai_robotics_third_party/cnpy
cd engineai_robotics_third_party
git clone https://github.com/rogersce/cnpy.git
```

### 步骤2: 修改CMakeLists.txt

```cmake
# 添加cnpy子目录
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../engineai_robotics_third_party/cnpy 
                 ${CMAKE_CURRENT_BINARY_DIR}/cnpy)

# 添加include目录
target_include_directories(rl_dance_runner
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../engineai_robotics_third_party/cnpy
)

# 链接库
target_link_libraries(rl_dance_runner
    PUBLIC
        cnpy
        z  # cnpy需要zlib
)
```

## 方法3: 使用FetchContent (推荐)

在 `src/runner/rl_dance/CMakeLists.txt` 的顶部添加：

```cmake
include(FetchContent)

# 下载并构建cnpy
FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(cnpy)

# 链接库
target_link_libraries(rl_dance_runner
    PUBLIC
        cnpy
        z  # zlib dependency
)
```

## 验证安装

编译后运行以下命令验证：

```bash
cd /home/lby/engineai_lby/repo/engineai_robotics
./build.sh

# 查看是否有链接错误
ldd build/src/runner/rl_dance/librl_dance_runner.so | grep cnpy
```

## 常见问题

### Q: 编译时找不到 zlib

```
undefined reference to `gzopen'
```

**A**: 需要安装zlib开发库：
```bash
sudo apt-get install zlib1g-dev
```

### Q: 链接时找不到cnpy符号

**A**: 确保在 `target_link_libraries` 中添加了 `cnpy` 和 `z`

### Q: 运行时找不到cnpy.so

**A**: 设置LD_LIBRARY_PATH：
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

或将cnpy安装路径添加到 `/etc/ld.so.conf.d/` 并运行 `sudo ldconfig`

