cmake_minimum_required(VERSION 3.23)
project(ImageHistogram C)

set(CMAKE_C_STANDARD 99)

add_executable(ImageHistogram src/main.c src/image_data.c)

target_include_directories(ImageHistogram PRIVATE src/include)