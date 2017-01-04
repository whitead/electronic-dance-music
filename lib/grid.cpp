#include "grid.h"

EDM::Grid* EDM::make_grid( int dim, 
		const double* min, 
		const double* max, 
		const double* bin_spacing, 
		const int* b_periodic, 
		int b_derivatives, 
		int b_interpolate) {
  switch(dim) {
  case 1:
    return new DimmedGrid<1>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  case 2:
    return new DimmedGrid<2>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  case 3:
    return new DimmedGrid<3>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  }

  return NULL;
}

EDM::Grid* EDM::read_grid( int dim, const std::string& filename, int b_interpolate) {
  switch(dim) {
  case 1:
    return new DimmedGrid<1>(filename, b_interpolate);
  case 2:
    return new DimmedGrid<2>(filename, b_interpolate);
  case 3:
    return new DimmedGrid<3>(filename, b_interpolate);
  }
  return NULL;
}


EDM::Grid* EDM::read_grid( int dim, const std::string& filename) {
  switch(dim) {
  case 1:
    return new DimmedGrid<1>(filename);
  case 2:
    return new DimmedGrid<2>(filename);
  case 3:
    return new DimmedGrid<3>(filename);
  }
  return NULL;
}

