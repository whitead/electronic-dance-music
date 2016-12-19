#ifndef GRID_CUH_
#define GRID_CUH_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "mpi.h"
#include <cuda_runtime.h>
#include "grid.h"
#include "edm.h"

#ifndef GRID_TYPE
#define GRID_TYPE 32
#endif //GRID_TYPE

namespace EDM{
  template<unsigned int DIM>
  class DimmedGridGPU : public DimmedGrid<DIM> {
    /** A DIM-dimensional grid for GPU use. Stores on 1D column-ordered array
     **/
  public:
    DimmedGridGPU(const double* min, 
	       const double* max, 
	       const double* bin_spacing, 
	       const int* b_periodic, 
	       int b_derivatives, 
	       int b_interpolate) : b_derivatives_(b_derivatives), b_interpolate_(b_interpolate), grid_(NULL), grid_deriv_(NULL) {

      size_t i;

      for(i = 0; i < DIM; i++) {
	min_[i] = min[i];
	max_[i] = max[i];
	b_periodic_[i] = b_periodic[i];

	grid_number_[i] = (int) ceil((max_[i] - min_[i]) / bin_spacing[i]);
	dx_[i] = (max_[i] - min_[i]) / grid_number_[i];
	//add one to grid points if 
	grid_number_[i] = b_periodic_[i] ? grid_number_[i] : grid_number_[i] + 1;
	//increment dx to compensate
	if(!b_periodic_[i])
	  max_[i] += dx_[i];
      }
      initialize();
    }

    size_t grid_size_;//total size of grid
    int b_derivatives_;//if derivatives are going to be used
    int b_interpolate_;//if interpolation should be used on the grid
    double* grid_;//the grid values
    double* grid_deriv_;//derivatives    
    double dx_[DIM];//grid spacing
    double min_[DIM];//grid minimum
    double max_[DIM];//maximum
    int grid_number_[DIM];//number of points on grid
    int b_periodic_[DIM];//if a dimension is periodic

  private:  

    /** This will actually allocate the arrays and perform any sublcass initialization
     *
     **/
    void initialize() {//this mallocs/cudamallocs our host/device grid_ & grid_deriv_ pointers
      size_t i;
      grid_size_ = 1;
      for(i = 0; i < DIM; i++)
	grid_size_ *= grid_number_[i];
      grid_ = (double *) calloc(DIM * grid_size_, sizeof(double));
//    cudaMalloc(&d_grid_, DIM * grid_size_ * sizeof(double));//need to make a d_grid_ pointer
      if(b_derivatives_) {
	grid_deriv_ = (double *) calloc(DIM * grid_size_, sizeof(double));
//      cudaMalloc(&d_grid_deriv_, DIM * grid_size_ * sizeof(double));//need to make a d_grid_deriv_
	if(!grid_deriv_) {
	  edm_error("Out of memory!!", "grid.cuh:initialize");	
	}
      }
    }


  
  };
}

#endif //GRID_CUH_
