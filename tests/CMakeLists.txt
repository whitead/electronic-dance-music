#Detect boost
find_package (Boost 1.36 COMPONENTS thread system filesystem unit_test_framework REQUIRED)  
include_directories (${Boost_INCLUDE_DIRS})

add_executable (edm_test edm_test)
target_link_libraries (edm_test
                      edm_lib
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
		      ${Boost_LIBRARIES})
