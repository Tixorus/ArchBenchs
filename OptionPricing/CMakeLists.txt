cmake_minimum_required(VERSION 3.10)
project(OptionPricing)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
include_directories(external)

set(SOURCES 
    src/european_analytical.cpp
    src/bermuda_RT.cpp 
    src/bermuda_SM.cpp 
    src/european_numerical.cpp 
    src/option_pricing_parallel.cpp
    external/toms462.cpp
)

find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    MKL::MKL
    OpenMP::OpenMP_CXX
)

