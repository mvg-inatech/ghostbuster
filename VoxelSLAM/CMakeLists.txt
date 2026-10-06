cmake_minimum_required(VERSION 3.5)
project(voxel_slam)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++14")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -g" )   # -g: line numbers in backtraces; keeps -O3 so timing is unchanged

# PointXYZIRGB carries 23 registered fields. PCL builds the field list with
# Boost.MPL, whose vector/list default to 20 elements, and going over that
# fails obscurely as "no type named 'type' in struct pcl::traits::fieldList".
# Raise the limit for our translation units (must be a multiple of 10, and must
# be defined before any boost header is included -- hence here, not in a source
# file).
add_definitions(-DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
                -DBOOST_MPL_LIMIT_VECTOR_SIZE=30
                -DBOOST_MPL_LIMIT_LIST_SIZE=30)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
  #livox_ros_driver
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED QUIET)
# find_package(Ceres REQUIRED)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs 
  DEPENDS EIGEN3 PCL GTSAM
  INCLUDE_DIRS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

add_executable(voxelslam src/voxelslam.cpp src/BTC.cpp)
target_link_libraries(voxelslam ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)







