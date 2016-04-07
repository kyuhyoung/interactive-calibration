cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME_STR interactive-calibration)

project (${PROJECT_NAME_STR})

option(USE_LAPACK "use lapack for SVD" ON)

if(USE_LAPACK)
    find_package(LAPACK)
    if(LAPACK_FOUND)
        find_file(LAPACK_HEADER "lapacke.h")
        if(LAPACK_HEADER_FUOND)
            add_definitions(-DUSE_LAPACK)
            link_libraries(${LAPACK_LIBRARIES})
        endif()
    endif()
else()
    set(LAPACK_LIBRARIES "")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_definitions(-std=c++11 -Wall -Wextra)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_definitions(-std=c++11 -stdlib=libc++)
endif()

find_package(OpenCV REQUIRED)

set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
 
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_INCLUDE_DIR}")

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${PROJECT_INCLUDE_DIR}/*.hpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${LAPACK_LIBRARIES})