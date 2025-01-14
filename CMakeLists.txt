cmake_minimum_required(VERSION 3.8)
project(ros_humble_tensorrt_bridge_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

include(CheckLanguage)
check_language(CUDA)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)


#find_library(cublas_LIBRARY cublas
#  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES lib64 lib
#  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
#  NO_DEFAULT_PATH
#)


if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message("${BoldRed}CUDA compiler not found${ColourReset}")
endif()

if(CUDA_FOUND)
  message("${BoldGreen}CUDA version: ${CUDA_VERSION_STRING}${ColourReset}")
  message("${Green}Using CUDA compiler: ${CMAKE_CUDA_COMPILER}${ColourReset}")
  message("${Green}CUDA libraries: ${CUDA_LIBRARIES}${ColourReset}")
  message("${Green}CUDA include directories: ${CUDA_INCLUDE_DIRS}${ColourReset}")
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARIES_DIRS})
else()
  message("${BoldRed}CUDA not found${ColourReset}")
endif()

if (OpenCV_FOUND)
  message("${BoldGreen}OpenCV version: ${OpenCV_VERSION}${ColourReset}")
else()
  message("${BoldRed}OpenCV not found${ColourReset}")
endif()

set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
include_directories(${TENSORRT_INCLUDE_DIR})

set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
set(TENSORRT_LIBS ${TENSORRT_LIB_DIR}/libnvinfer.so ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so ${TENSORRT_LIB_DIR}/libnvonnxparser.so)

message("${Green}TensorRT include directory: ${TENSORRT_INCLUDE_DIR}${ColourReset}")
message("${Green}TensorRT library directory: ${TENSORRT_LIB_DIR}${ColourReset}")
message("${Green}TensorRT libraries: ${TENSORRT_LIBS}${ColourReset}")

include_directories(include)

set(library_name tensorrt_bridge)
set(executable_name main)

add_library(${library_name} SHARED src/tensorrt_bridge.cpp)
target_link_libraries(${library_name} ${TENSORRT_LIBS} ${CUDA_INCLUDE_DIRS})
ament_target_dependencies(${library_name} rclcpp)

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(${executable_name} src/main.cpp)
target_link_libraries(${executable_name} ${library_name})

install(TARGETS ${executable_name} RUNTIME DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
