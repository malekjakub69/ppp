#
# @file   CMakeLists.txt
#
# @author David Bayer \n
#         Faculty of Information Technology \n
#         Brno University of Technology \n
#         ibayer@fit.vutbr.cz
#
# @brief  CMakeLists for lab1
#
# @date   14 February 2024, 14:00 (created) \n
#

cmake_minimum_required(VERSION 3.20)

# project name
project(lab1 LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find and setup MPI package
set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

# compile p2p
add_executable(p2p p2p.cpp)
target_link_libraries(p2p MPI::MPI_CXX)

# compile solution if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/p2p_solution.cpp")
  add_executable(p2p_solution p2p_solution.cpp)
  target_link_libraries(p2p_solution MPI::MPI_CXX)
endif()
