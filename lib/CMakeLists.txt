add_definitions(-DEDM_SERIAL -DEDM_GPU_MODE)
#add_library (edm_lib SHARED edm_bias grid gaussian_grid edm)
cuda_add_library(edm_cuda_lib edm_bias.cu edm_bias.cuh grid.cuh grid.cpp gaussian_grid.cuh edm.cpp edm.h)
