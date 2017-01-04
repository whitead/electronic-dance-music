#include "gaussian_grid_gpu.cuh"

EDM::GaussGrid* EDM::make_gauss_grid_gpu( int dim, 
			   const double* min, 
			   const double* max, 
			   const double* bin_spacing, 
			   const int* b_periodic, 
			   int b_interpolate,
			   const double* sigma) {
  switch(dim) {
  case 1:
    return new DimmedGaussGridGPU<1>(min, max, bin_spacing, b_periodic, b_interpolate, sigma);
  case 2:
    return new DimmedGaussGridGPU<2>(min, max, bin_spacing, b_periodic, b_interpolate, sigma);
  case 3:
    return new DimmedGaussGridGPU<3>(min, max, bin_spacing, b_periodic, b_interpolate, sigma);
  }

  return NULL;
}


EDM::GaussGrid* EDM::read_gauss_grid_gpu( int dim, const std::string& filename, const double* sigma) {
  switch(dim) {
  case 1:
    return new DimmedGaussGridGPU<1>(filename, sigma);
  case 2:
    return new DimmedGaussGridGPU<2>(filename, sigma);
  case 3:
    return new DimmedGaussGridGPU<3>(filename, sigma);
  }
  return NULL;
}
