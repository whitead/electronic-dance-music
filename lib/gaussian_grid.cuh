#ifndef GPU_GAUSS_GRID_H_
#define GPU_GAUSS_GRID_H_

#include "gaussian_grid.h"
#include "grid.cuh"
#include "edm.h"
#include <string>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#ifndef GAUSS_SUPPORT
#define GAUSS_SUPPORT 8.0 // sigma^2 considered for gaussian
#endif
#ifndef BC_TABLE_SIZE
#define BC_TABLE_SIZE 65536 //boundary correction function look up size
#endif
#ifndef BC_MAR
#define BC_MAR 2.0
#endif
#ifndef BC_CORRECTION
#define BC_CORRECTION
#endif


namespace EDM{

/**
 * This class uses a compositional ("has-a") relationship with Grid. This is the 
 * interface and DimmedGaussGrid has the dimension template parameter
 **/
  class GPUGaussGrid : public GaussGrid {
    /**
     * Retruns the integrated amount of bias added to the system
     **/
  public:
    virtual ~GPUGaussGrid() {};
    double add_value(const double* x, double height) = 0;
//  __device__ virtual double add_hills_gpu(const double* buffer, const size_t hill_number, char hill_type, double *grid_);
  };

  template<unsigned int DIM>
  class GPUDimmedGaussGrid : public DimmedGaussGrid<DIM>{
    /** A class for treating grids that have gaussians on it 
     *
     *
     **/
  public:
    GPUDimmedGaussGrid(const double* min, 
		       const double* max, 
		       const double* bin_spacing, 
		       const int* b_periodic, 
		       int b_interpolate,
		       const double* sigma) : DimmedGaussGrid<DIM>(min, max, bin_spacing, b_periodic, b_interpolate, sigma) {}

    /**
     * Rebuild from a file. Files don't store sigma, so it must be set again.
     **/
    GPUDimmedGaussGrid(const std::string& filename, const double* sigma) : DimmedGaussGrid<DIM>( filename, sigma) {}
  
    ~GPUDimmedGaussGrid() {
      //nothing
    }

  };
  
/**
 * Used to avoid template constructors
 **/
  GaussGrid* make_gpu_gauss_grid(unsigned int dim, 
			     const double* min, 
			     const double* max, 
			     const double* bin_spacing, 
			     const int* b_periodic, 
			     int b_interpolate,
			     const double* sigma);

/**
   p * Used to avoid template constructors
**/
  GaussGrid* read_gpu_gauss_grid(unsigned int dim, const std::string& filename, const double* sigma);

}
#endif //GPU_GAUSS_GRID_H_
