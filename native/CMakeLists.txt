cmake_minimum_required(VERSION 3.20)
project(autodancer_native LANGUAGES C)

add_library(autodancer_native SHARED autodancer_native.c)
target_compile_definitions(autodancer_native PRIVATE UNICODE _UNICODE)
set_target_properties(autodancer_native PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    PREFIX ""
    OUTPUT_NAME "autodancer_native"
)
