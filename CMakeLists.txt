cmake_minimum_required(VERSION 3.24.1 FATAL_ERROR)
project(torchTest)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Add the path to opencv here
find_package(OpenCV REQUIRED PATHS C:/Users/sasso/Downloads/ai-lec/opencv/build )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(torchTest torchExample.cpp 
			 src/TorchModel.cpp
			 src/CRAFT.cpp
			 src/CRNN.cpp)
target_include_directories(torchTest PUBLIC "include/")
target_link_libraries(torchTest ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${MKL})
set_property(TARGET torchTest PROPERTY CXX_STANDARD 14)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT torchTest)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torchTest
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torchTest>)
endif (MSVC)