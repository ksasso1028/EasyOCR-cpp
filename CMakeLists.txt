cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(torchTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add libtorch folder if built from dockerfile
if (DEFINED ENV{build_from_docker_file})
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    list(APPEND CMAKE_PREFIX_PATH ${PARENT_DIR}/thirdparty/libtorch)
endif()
find_package(Torch REQUIRED)

# Find OPENCV automatically if built from dockerfile
if (DEFINED ENV{build_from_docker_file})
    find_package(OpenCV 4 REQUIRED)
    find_package(Threads REQUIRED)
# Else force the path to opencv here if executed on Windows
else ()
    find_package(OpenCV REQUIRED PATHS C:/Users/sasso/Downloads/new/opencv/build )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Get all character text files  from the 'lang' directory
file(GLOB LANG_FILES "lang/*")

file(GLOB MODELS "models/*")


# Set the source file path
set(SOURCE_FILE_PATH "test.jpg")

# Set the destination file path in the binary directory
set(DEST_FILE_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Copy the file to the binary directory
configure_file(${SOURCE_FILE_PATH} ${DEST_FILE_PATH} COPYONLY)

# Copy files obtained from FILE_PATHS_GLOB_1
foreach(FILE_PATH ${LANG_FILES})
    get_filename_component(FILE_NAME ${FILE_PATH} NAME)  # Get the file name
    file(COPY ${FILE_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

# Copy files obtained from FILE_PATHS_GLOB_2
foreach(FILE_PATH ${MODELS})
    get_filename_component(FILE_NAME ${FILE_PATH} NAME)  # Get the file name
    file(COPY ${FILE_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

add_executable(torchTest torchExample.cpp 
			 src/TorchModel.cpp
			 src/CRAFT.cpp
			 src/CRNN.cpp)
target_include_directories(torchTest PUBLIC "include/")
target_link_libraries(torchTest ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${MKL})
set_property(TARGET torchTest PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT torchTest)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torchTest
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torchTest>)
endif (MSVC)