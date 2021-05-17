cmake_minimum_required(VERSION 3.13)
project(CAP-PLAB2021 LANGUAGES C CUDA)

set(SOURCES common.c nn-main.cu)

add_executable(CAP-PLAB2021.exe ${SOURCES})
#set_property(TARGET CAP-PLAB2021.exe PROPERTY C_STANDARD 99)

#target_link_libraries(CAP-PLAB2021.exe)
