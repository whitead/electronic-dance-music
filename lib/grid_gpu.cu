#include "grid_gpu.cuh"

EDM::Grid* EDM::make_grid_gpu( int dim, 
		const edm_data_t* min, 
		const edm_data_t* max, 
		const edm_data_t* bin_spacing, 
		const int* b_periodic, 
		int b_derivatives, 
		int b_interpolate) {
  switch(dim) {
  case 1:
    return new DimmedGridGPU<1>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  case 2:
    return new DimmedGridGPU<2>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  case 3:
    return new DimmedGridGPU<3>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  }

  return NULL;
}

EDM::Grid* EDM::read_grid_gpu( int dim, const std::string& filename, int b_interpolate) {
  printf("read_grid_gpu was called with dimension %i\n", dim);
  switch(dim) {
  case 1:
    return new DimmedGridGPU<1>(filename, b_interpolate);
  case 2:
    return new DimmedGridGPU<2>(filename, b_interpolate);
  case 3:
    return new DimmedGridGPU<3>(filename, b_interpolate);
  }
  return NULL;
}


EDM::Grid* EDM::read_grid_gpu( int dim, const std::string& filename) {
  printf("read_grid_gpu was called with dimension %i\n", dim);
  switch(dim) {
  case 1:
    return new DimmedGridGPU<1>(filename);
  case 2:
    return new DimmedGridGPU<2>(filename);
  case 3:
    return new DimmedGridGPU<3>(filename);
  }
  return NULL;
}