#include "gaussian_grid.h"

EDM::GaussGrid* EDM::make_gauss_grid(unsigned int dim, 
			   const double* min, 
			   const double* max, 
			   const double* bin_spacing, 
			   const int* b_periodic, 
			   int b_interpolate,
			   const double* sigma) {
  switch(dim) {
  case 1:
    return new DimmedGaussGrid<1>(min, max, bin_spacing, b_periodic, b_interpolate, sigma);
  case 2:
    return new DimmedGaussGrid<2>(min, max, bin_spacing, b_periodic, b_interpolate, sigma);
  case 3:
    return new DimmedGaussGrid<3>(min, max, bin_spacing, b_periodic, b_interpolate, sigma);
  }

  return NULL;
}


EDM::GaussGrid* EDM::read_gauss_grid(unsigned int dim, const std::string& filename, const double* sigma) {
  switch(dim) {
  case 1:
    return new DimmedGaussGrid<1>(filename, sigma);
  case 2:
    return new DimmedGaussGrid<2>(filename, sigma);
  case 3:
    return new DimmedGaussGrid<3>(filename, sigma);
  }
  return NULL;
}
