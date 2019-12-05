cmake_minimum_required(VERSION 3.0)

project(classification)

set(LG_PATH $ENV{LG_PATH})
if(NOT LG_PATH)
  message( FATAL_ERROR "LG_PATH is not defined" )
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -msse2 -msse4 -D_GLIBCXX_USE_CXX11_ABI=0")

add_subdirectory(${LG_PATH} loadgen)

find_package(OpenCV REQUIRED)
set(TF_PATH $ENV{TF_PATH})
set(CMAKE_CXX_FLAGS "-Wl,--no-undefined -fPIC -std=c++11 -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0")
execute_process(COMMAND bash "-c" "${PROJECT_SOURCE_DIR}/tf_version.sh" OUTPUT_VARIABLE TF_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash "-c" "${PROJECT_SOURCE_DIR}/python_site.sh" OUTPUT_VARIABLE PYTHON_SITE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT TF_PATH)
  message( FATAL_ERROR "TF_PATH is not defined" )
endif()

get_filename_component(TF_NAME ${TF_PATH} NAME)

include_directories(${TF_PATH}
                    ${TF_PATH}/bazel-${TF_NAME}/external/eigen_archive
                    ${TF_PATH}/bazel-${TF_NAME}/external/protobuf_archive/src
                    ${TF_PATH}/bazel-${TF_NAME}/external/com_google_absl
                    ${TF_PATH}/bazel-genfiles
                    ${LG_PATH}/bindings
                    ${PYTHON_SITE_PATH}/ratelnn/include)

link_directories(${TF_PATH}/bazel-bin/tensorflow
                 ${PYTHON_SITE_PATH}/ratelnn/lib
                 ${PYTHON_SITE_PATH}/tensorflow)

message( "\n------ list all include directories ------")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
message( "" )

message( "\n------ list all link directories ------")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY LINK_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
message( "" )

add_executable(classification classification.cc op.cc image_loader.cc test_harness.cc)
target_link_libraries(classification mlperf_loadgen tensorflow_cc ratelrt dl tensorflow_framework glog jpeg)
target_link_libraries(classification ${OpenCV_LIBS} )

add_custom_command(TARGET classification POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:classification> ${PROJECT_SOURCE_DIR}
)
