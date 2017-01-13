#need to update this once the whole header file situation is sorted out
add_definitions(-DEDM_SERIAL)
#cuda_add_library (edm_lib SHARED edm_bias grid gaussian_grid edm )
cuda_add_library(edm_lib edm_bias_gpu.cuh edm_bias_gpu.cu edm_bias.cpp edm_bias.h  grid.cpp grid.h grid_gpu.cuh gaussian_grid.cpp gaussian_grid.h edm.cpp edm.h gaussian_grid_gpu.cu gaussian_grid_gpu.cuh)
set_target_properties(edm_lib PROPERTIES LINK_FLAGS -m64)





