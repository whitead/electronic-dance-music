#Detect boost
find_package (Boost 1.36 COMPONENTS system filesystem unit_test_framework REQUIRED)  
include_directories (${Boost_INCLUDE_DIRS})
set (TEST_GRID_SRC grids)
set (TEST_EDM_SRC edm)
add_definitions(-DTEST_GRID_SRC=${TEST_GRID_SRC})
add_definitions(-DTEST_EDM_SRC=${TEST_EDM_SRC})

add_executable (edm_test edm_test)
target_link_libraries (edm_test
                       edm_lib
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
