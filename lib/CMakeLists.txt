#need to update this once the whole header file situation is sorted out
add_definitions(-DEDM_SERIAL -DEDM_GPU_MODE)
#cuda_add_library (edm_lib SHARED edm_bias grid gaussian_grid edm )
cuda_add_library(edm_lib edm_bias.cuh edm_bias.cu edm_bias.cpp edm_bias.h  grid.cpp grid.h grid.cuh gaussian_grid.cpp gaussian_grid.h edm.cpp edm.h gaussian_grid.cu gaussian_grid.cuh)
set_target_properties(edm_lib PROPERTIES LINK_FLAGS -m64)





