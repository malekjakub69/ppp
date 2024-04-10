#
# @file   CMakeLists.txt
#
# @author David Bayer \n
#         Faculty of Information Technology \n
#         Brno University of Technology \n
#         ibayer@fit.vutbr.cz
#
# @brief  CMakeLists for lab7
#
# @date   04 April 2024, 17:00 (created) \n
#

cmake_minimum_required(VERSION 3.20)

# project name
project(lab7 LANGUAGES C CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find and setup MPI package
set(MPI_CXX_SKIP_MPICXX ON)
find_package(MPI REQUIRED COMPONENTS CXX)

# find and setup HDF5 package
set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED COMPONENTS C)

if(NOT ${HDF5_IS_PARALLEL})
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(FATAL_ERROR "This lab must be linked to HDF5 library with parallel (mpi) support, on Windows you need to build it from source")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Mac OS X")
    message(FATAL_ERROR "This lab must be linked to HDF5 library with parallel (mpi) support, try installing it using `brew install hdf5-mpi`")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This lab must be linked to HDF5 library with parallel (mpi) support, try installing it using `sudo apt install libhdf5-openmpi-dev`")
  else()
    message(FATAL_ERROR "This lab must be linked to HDF5 library with parallel (mpi) support")
  endif()
endif()

# check if hdf5 tools are installed
find_program(H5LS_EXECUTABLE NAMES h5ls)
find_program(H5DUMP_EXECUTABLE NAMES h5dump)

if(NOT H5LS_EXECUTABLE OR NOT H5DUMP_EXECUTABLE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(FATAL_ERROR "This lab requires h5ls and h5dump utilities, on Windows you need to build them from source (because of MPI support)")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Mac OS X")
    message(FATAL_ERROR "This lab requires h5ls and h5dump utilities, try installing them using `brew install hdf5-mpi`")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This lab requires h5ls and h5dump utilities, try installing them using `sudo apt install hdf5-tools`")
  else()
    message(FATAL_ERROR "This lab requires h5ls and h5dump utilities")
  endif()
endif()

# copy Matrix-File.h5 to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Matrix-File.h5 DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# compile hdf5_assignment
add_executable(hdf5_assignment ${CMAKE_CURRENT_SOURCE_DIR}/hdf5.cpp)
set_target_properties(hdf5_assignment PROPERTIES OUTPUT_NAME "hdf5")
target_link_libraries(hdf5_assignment HDF5::HDF5 MPI::MPI_CXX)

# compile solution if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/hdf5_solution.cpp")
  add_executable(hdf5_solution hdf5_solution.cpp)
  target_link_libraries(hdf5_solution HDF5::HDF5 MPI::MPI_CXX)
endif()

# make pack_assignment target
add_custom_target(pack_assignment
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/hdf5.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/Matrix-File.h5"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/load_modules.sh"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# make pack_solution target
add_custom_target(pack_solution
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/hdf5.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/hdf5_solution.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/Matrix-File.h5"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/load_modules.sh"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
