#Detect boost
#FOR BLUEHIVE
#set(BOOST_INCLUDEDIR /software/boost/1.66.0/b1/include/boost)
#set(BOOST_ROOT /software/boost/1.66.0/b1/)
#find_package (Boost 1.66 COMPONENTS system timer filesystem unit_test_framework REQUIRED)  
#FOR MY MACHINE
find_package (Boost 1.55 COMPONENTS system timer filesystem unit_test_framework REQUIRED)  
include_directories (${Boost_INCLUDE_DIRS})
message("The Boost include dirs are at ${Boost_INCLUDE_DIRS}")
set (TEST_GRID_SRC grids)
set (TEST_EDM_SRC edm)

include_directories (/usr/include/ /usr/local/include/ /usr/local/cuda/include/ ../lib/ )
add_definitions(-DTEST_GRID_SRC=${TEST_GRID_SRC})
add_definitions(-DTEST_EDM_SRC=${TEST_EDM_SRC})
#add_definitions(-DUSE_DOUBLES)

cuda_add_executable (edm_gpu_test edm_gpu_test.cu)
cuda_add_executable (edm_timing edm_timing_test.cu)
add_executable (edm_test edm_test.cpp)
target_compile_definitions(edm_test PUBLIC "-DEDM_GPU_MODE=0")
target_link_libraries (edm_test
                       edm_gpu
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
		      ${Boost_LIBRARIES}
		      ${MPI_CXX_LIBRARIES})
target_compile_definitions(edm_gpu_test PUBLIC "-DEDM_GPU_MODE=1")
target_link_libraries (edm_gpu_test
                       edm_gpu
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
		      ${Boost_LIBRARIES}
		      ${MPI_CXX_LIBRARIES})
target_link_libraries (edm_timing
                       edm_gpu
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
		      ${Boost_LIBRARIES}
		      ${MPI_CXX_LIBRARIES})



file (COPY 1.grid DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${TEST_GRID_SRC}")
file (COPY 2.grid DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${TEST_GRID_SRC}")
file (COPY 3.grid DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${TEST_GRID_SRC}")
file (COPY read_test.edm DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${TEST_EDM_SRC}")
file (COPY sanity.edm DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${TEST_EDM_SRC}")
