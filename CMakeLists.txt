cmake_minimum_required(VERSION 3.1)


project(main LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)

# CUDA
find_package(CUDA 11.8 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT
# set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
# set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)

include_directories(/usr/local/include/opencv4/)
set(OPENCV_LIB_PATH  /usr/local/lib)
file(GLOB OPENCV_LIBS "${OPENCV_LIB_PATH}/*.so")

include_directories(/home/TensorRT-8.6.1.6/include)
set(TENSORRT_LIB_PATH  /home/TensorRT-8.6.1.6/lib)
file(GLOB TENSORRT_LIB "${TENSORRT_LIB_PATH}/*.so")

list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_LIB_DIR}
        )

include_directories(${INCLUDE_DIRS})

add_definitions(-DDEBUG)  # 启用调试输出




# 添加目标文件
set(SRC_DIR /home/sam2TRT-main/src)
add_executable(${PROJECT_NAME}
        main.cpp
        ${SRC_DIR}/sam2.cpp
        )

# 将库路径添加到目标链接
target_link_directories(${PROJECT_NAME} PUBLIC ${ALL_LIBS})

# 链接相关的库
target_link_libraries(${PROJECT_NAME}
        PRIVATE nvinfer 
        PRIVATE nvinfer_plugin 
        ${TENSORRT_LIB} 
        ${OPENCV_LIBS}
        ${CUDA_LIBRARIES}
)




# # 设置运行时库路径，确保可以找到 .so 文件
# set(CMAKE_INSTALL_RPATH /home/xc/wineYolov8Trt_base/build)


# 其他的可能的配置
# install(TARGETS ${PROJECT_NAME} DESTINATION /path/to/install)
