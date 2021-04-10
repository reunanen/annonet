cmake_minimum_required(VERSION 3.13)
project(annonet_rest-infer)

add_subdirectory(dlib dlib_build)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-fopenmp")

include_directories(
  .
  dlib
  rapidjson/include
  )

file (GLOB Lib_sources
  "annonet.cpp"
  "annonet_infer.cpp"
  "annonet_parse_anno_classes.cpp"
  "dlib-dnn-pimpl-wrapper/*.cpp"
  "cpp-read-file-in-memory/read-file-in-memory.cpp"
  "tiling/tiling.cpp"
  )

add_executable(annonet_train ${Lib_sources} annonet_train_main.cpp)
add_executable(annonet_infer ${Lib_sources} annonet_infer_main.cpp)

target_link_libraries(annonet_train curl dlib::dlib)
target_link_libraries(annonet_infer curl dlib::dlib)
