cmake_minimum_required(VERSION 2.8)
project(dense_monocular)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")
# OpenCV
#set(OpenCV_DIR "/home/gi/Downloads/opencv-3.4.5")
find_package(OpenCV 3.4.5 REQUIRED)
find_package(Glog REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
set(Sophus_DIR "../3rdparty/Sophus")
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
	${GLOG_LIBRARIES}
        glog::glog
)

#add_executable(dense_mapping dense_mapping.cpp)
#target_link_libraries(dense_mapping ${THIRD_PARTY_LIBS})
add_executable(dense_mapping_gpu dense_mapping_gpu.cu)
target_link_libraries(dense_mapping_gpu ${THIRD_PARTY_LIBS})
