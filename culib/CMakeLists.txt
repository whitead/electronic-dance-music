add_definitions(-DEDM_SERIAL)
#add_library (edm_lib SHARED edm_bias grid gaussian_grid edm)
cuda_add_library(edm_cuda_lib edm_bias.cu edm_bias.cuh grid.cu grid.cuh gaussian_grid.cu gaussian_grid.cuh edm.cu edm.cuh)
set_target_properties(edm_cuda_lib PROPERTIES LINK_FLAGS -m64)










