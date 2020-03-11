cmake_minimum_required(VERSION 2.4)
project(furnari_flow)

# OpenCV Config
find_package( OpenCV REQUIRED )

message(STATUS "    version: ${OpenCV_VERSION}")
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

ADD_DEFINITIONS(-ggdb -O3 -Wall -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -fbranch-probabilities)

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

################################

ADD_EXECUTABLE(
  compute_flow
  src/compute_flow.cpp
  ${SRCS}
)

TARGET_LINK_LIBRARIES(
  compute_flow
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)

################################

foreach(dir ${DIRS})
  FILE(GLOB files ${dir}/*.h*)
  LIST(APPEND h_files ${files})
endforeach(dir)
add_custom_target(TAGS etags --members -l c++ ${h_files} ${SRCS})
