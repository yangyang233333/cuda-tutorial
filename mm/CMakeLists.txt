cmake_minimum_required(VERSION 3.16)
project(mm CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(/usr/local/cuda-11/include)
file(GLOB SRC ./*.cu)

add_executable(${PROJECT_NAME} ${SRC})
