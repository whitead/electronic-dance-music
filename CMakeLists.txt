cmake_minimum_required (VERSION 2.6)

#project info
project (WordModel)
set (WordModel_VERSION_MAJOR 0)
set (WordModel_VERSION_MINOR 1)
set (WordModel_DESCRIPTION "Experiment directed simulation implementation")

#compilation flags
#set (CXX_STD_VERSION "-std=c++11 -g")
set (CXX_STD_VERSION "-std=c++11 -O0 -g")
add_definitions (${CXX_STD_VERSION})


#set-up directories
include_directories ("${PROJECT_SOURCE_DIR}/src")


#compile src
add_subdirectory ("lib")

#compile unit tests
add_subdirectory ("tests")

