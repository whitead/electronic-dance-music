cmake_minimum_required (VERSION 2.6)

#project info
project (WordModel)
set (WordModel_VERSION_MAJOR 0)
set (WordModel_VERSION_MINOR 1)
set (WordModel_DESCRIPTION "Experiment directed simulation implementation")
set (LAMMPS_SRC "/home/whitead/Documents/lammps/src")


#compilation flags
#set (CXX_STD_VERSION "-std=c++11 -g")
set (CXX_DEBUG "-O0 -g -Wall -Wno-sign-compare")
add_definitions (${CXX_DEBUG})

find_package(MPI REQUIRED) 
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


#set-up directories
include_directories ("${PROJECT_SOURCE_DIR}/lib")


#compile src
add_subdirectory ("lib")

#compile unit tests
add_subdirectory ("tests")

#compile python library
add_subdirectory ("python")

file(GLOB LammpsFiles ${CMAKE_SOURCE_DIR}/lib/*.cpp
                      ${CMAKE_SOURCE_DIR}/lib/*.h
                      ${CMAKE_SOURCE_DIR}/*.h
                      ${CMAKE_SOURCE_DIR}/*.cpp)

add_custom_target(patch)
foreach(f ${LammpsFiles})
  add_custom_command(TARGET patch PRE_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                         copy ${f} ${LAMMPS_SRC})
endforeach()