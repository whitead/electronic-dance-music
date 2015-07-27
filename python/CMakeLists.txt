find_package(Boost 1.45.0 COMPONENTS python REQUIRED)
find_package(PythonLibs)
if(Boost_FOUND AND PYTHONLIBS_FOUND)
  include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_PATH})
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)

  add_library(edm_python SHARED edm_python.cxx edm_bias_py)
  target_link_libraries(edm_python edm_lib ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})

  add_custom_target(python_package)  
  set(EDM_Package edm)

  file (COPY __init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  
  add_custom_command(TARGET python_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EDM_Package})
  add_custom_command(TARGET python_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename __init__.py ${EDM_Package}/__init__.py)
  add_custom_command(TARGET python_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename libedm_python.so ${EDM_Package}/libedm_python.so)
elseif(NOT Boost_FOUND)
  message(FATAL_ERROR "Unable to find correct Boost version and Python Lib")
endif()

