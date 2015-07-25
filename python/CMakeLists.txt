find_package(Boost 1.45.0 COMPONENTS python REQUIRED)
find_package(PythonLibs)
if(Boost_FOUND AND PYTHONLIBS_FOUND)
  include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_PATH})
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)

  add_library(edm_python SHARED edm_python.cxx)
  target_link_libraries(edm_python ${Boost_LIBRARIES} edm_lib ${MPI_CXX_LIBRARIES})
elseif(NOT Boost_FOUND)
  message(FATAL_ERROR "Unable to find correct Boost version and Python Lib")
endif()

