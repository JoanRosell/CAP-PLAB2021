cmake_minimum_required(VERSION 3.13)
project(CAP-PLAB2021)

set(SOURCES common.c nn-main.c)
set(HEADERS common.h)

add_executable(CAP-PLAB2021.exe ${SOURCES} ${HEADERS})
set_property(TARGET CAP-PLAB2021.exe PROPERTY C_STANDARD 99)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -g3 -Wall")
target_link_libraries(CAP-PLAB2021.exe PUBLIC m OpenMP::OpenMP_C)
