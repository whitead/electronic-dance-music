#need to update this once the whole header file situation is sorted out
add_definitions(-DEDM_SERIAL)# -DUSE_DOUBLES)#don't define USE_DOUBLES if your GPU can't use doubles
#cuda_add_library (edm_lib SHARED edm_bias grid gaussian_grid edm )
cuda_add_library(edm_gpu SHARED grid_gpu.cu edm_bias_gpu.cu edm_bias.cpp  grid.cpp gaussian_grid.cpp edm.cpp gaussian_grid_gpu.cu)
target_link_libraries(edm_gpu PRIVATE cudart)
set_target_properties(edm_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out LINK_FLAGS -m64)
