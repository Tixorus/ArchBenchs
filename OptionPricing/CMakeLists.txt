cmake_minimum_required(VERSION 3.10)
project(OptionPricing)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
include_directories(external)

set(SOURCES 
    src/analytical.cpp
    src/bermuda.cpp 
    src/numerical.cpp 
    src/option-pricing-parallel.cpp
    external/toms462.cpp
)

find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    MKL::MKL
    OpenMP::OpenMP_CXX
)

