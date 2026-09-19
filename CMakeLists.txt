cmake_minimum_required(VERSION 3.10)

project(neural_net C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(neural_net
    src/main.c
    src/neuron.c
    src/neural_layer.c
    src/neural_net.c
)

target_link_libraries(neural_net PRIVATE m)