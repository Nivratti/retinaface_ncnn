cmake_minimum_required(VERSION 3.9)
set( CMAKE_TOOLCHAIN_FILE "/content/vcpkg/scripts/buildsystems/vcpkg.cmake" )

project (liveness)

add_definitions(-std=c++11)
add_definitions("-Wall")


## Statically link-- os specific ncnn
## more info: https://github.com/Tencent/ncnn/releases
## cmake link -- https://github.com/Tencent/ncnn/wiki/use-ncnn-with-own-project

# set install path of ncnn on your system
set(NCNN_INSTALL_DIR "/usr/local/c++/ncnn/build/install")

## install dir will be created after running 'make install' command in ncnn build dir
set(ncnn_DIR "${NCNN_INSTALL_DIR}/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)


# =============================================================================
# make executable for testing purpose
# -----------------------------------------------------------------------------
add_executable(retinaface
	FaceDetector.cpp 
	FaceDetector.h 
	main_executable.cpp
)

# link ncnn // ncnn already defined above -- just use it
target_link_libraries( retinaface  ncnn )

# Find Package
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  target_link_libraries( retinaface ${OpenCV_LIBS} )
endif()

# link opencv
# target_link_libraries( retinaface ${OpenCV_LIBS} )