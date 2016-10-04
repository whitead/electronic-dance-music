find_package(Boost 1.45.0 COMPONENTS python REQUIRED)
find_package(PythonLibs)
set(EDM_PACKAGE edm)
if(Boost_FOUND AND PYTHONLIBS_FOUND)
  include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_PATH})
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)

  file (COPY ${EDM_PACKAGE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file (COPY setup.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  
  add_library(edm_python SHARED ${EDM_PACKAGE}/edm_python.cxx ${EDM_PACKAGE}/edm_bias_py)
  set_target_properties(edm_python PROPERTIES COMPILE_FLAGS -I${PROJECT_SOURCE_DIR}/lib LINK_FLAGS -I${PROJECT_SOURCE_DIR}/lib PREFIX "${EDM_PACKAGE}/lib")
  target_link_libraries(edm_python edm_cuda_lib ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
  
elseif(NOT Boost_FOUND)
  message(FATAL_ERROR "Unable to find correct Boost version and Python Lib")
endif()

