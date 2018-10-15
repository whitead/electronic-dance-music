cmake_minimum_required (VERSION 2.8)



#project info
project (WordModel)
set (WordModel_VERSION_MAJOR 0)
set (WordModel_VERSION_MINOR 1)
set (WordModel_DESCRIPTION "Experiment directed metadynamics implementation")
set (LAMMPS_SRC "/home/rainier/lammps/src")

#compilation flags
set (CXX_STD_VERSION "-std=c++11 -g")
set (CMAKE_VERBOSE_MAKEFILE 0) # Use 1 for debugging, 0 for release
set(CMAKE_CXX_COMPILER g++)
set (CXX_DEBUG "-O0 -g -m64 -std=c++11 -Wall -Wno-sign-compare -Wno-reorder")
add_definitions (${CXX_DEBUG})
#adding coverage flags
set(CMAKE_C_FLAGS "-m64 -fprofile-arcs -ftest-coverage" )
set(CMAKE_CXX_FLAGS "-m64 -fprofile-arcs -ftest-coverage")


#the CUDA stuff
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
#set(CUDA_NVCC_FLAGS "-std=c++11 -I/usr/include/ -I/usr/local/include/ -I/usr/local/cuda/include/ -g -G -lineinfo -cudart shared -Xlinker -rpath,"."")
find_package(CUDA QUIET REQUIRED) #this is for getting CUDA cross-compilation
#set(CUDA_NVCC_FLAGS -m64)
include_directories (/usr/include/ /usr/local/include/ /usr/local/cuda/include/ )


find_package(MPI REQUIRED) 
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


#set-up directories
include_directories ("${PROJECT_SOURCE_DIR}/culib")


#compile src
add_subdirectory ("lib")

#compile unit tests
add_subdirectory ("tests")

#compile python library
#add_subdirectory ("python")

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
