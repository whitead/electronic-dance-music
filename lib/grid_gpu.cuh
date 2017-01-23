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
  template< int DIM>
  class DimmedGridGPU : public DimmedGrid<DIM> {
    /** A DIM-dimensional grid for GPU use. Stores on 1D column-ordered array
     **/
  public:
    DimmedGridGPU(const double* min, 
	       const double* max, 
	       const double* bin_spacing, 
	       const int* b_periodic, 
	       int b_derivatives, 
		  int b_interpolate) : DimmedGrid<DIM> ( min, 
	        max, 
	        bin_spacing, 
	        b_periodic, 
	        b_derivatives, 
		b_interpolate){

    }


    /**
     * Constructor from file, with interpolation specified
     **/
    DimmedGridGPU(const std::string& input_grid, int b_interpolate) {
      b_interpolate_ = b_interpolate;
      this->read(input_grid);
    }

    /**
     * Constructor from grid file
     **/
    DimmedGridGPU(const std::string& input_grid){
      this->read(input_grid);
    }

    ~DimmedGridGPU() {
      if(grid_ != NULL){
	cudaFree(grid_);
	grid_ = NULL;//need to do this so DimmedGrid's destructor functions properly
      }
	
      if(grid_deriv_ != NULL){
	cudaFree(grid_deriv_);
	grid_deriv_ = NULL;
      }
	
    }

    virtual double get_value(const double* x) const{
      cudaDeviceSynchronize();
      if(!(this->in_grid(x))){
	return 0;
      }
      if(b_interpolate_ && b_derivatives_) {
	double temp[DIM];
	return this->get_value_deriv(x, temp);
      }

      size_t index[DIM];
      this->get_index(x, index);
      return grid_[this->multi2one(index)];
    }

//need to tell compiler where to find these since we have a derived templated class.
    using DimmedGrid<DIM>::grid_size_;
    using DimmedGrid<DIM>::b_derivatives_;
    using DimmedGrid<DIM>::b_interpolate_;
    using DimmedGrid<DIM>::grid_;
    using DimmedGrid<DIM>::grid_deriv_;
    using DimmedGrid<DIM>::dx_;
    using DimmedGrid<DIM>::min_;
    using DimmedGrid<DIM>::max_;
    using DimmedGrid<DIM>::grid_number_;
    using DimmedGrid<DIM>::b_periodic_;

  private:  

    /** This will actually allocate the arrays and perform any sublcass initialization
     *
     **/
    virtual void initialize() {//this cudamallocs our device grid_ & grid_deriv_ pointers
      size_t i;
      grid_size_ = 1;
      for(i = 0; i < DIM; i++)
	grid_size_ *= grid_number_[i];
      cudaMallocManaged(&grid_, grid_size_ * sizeof(double));
      if(b_derivatives_) {
	cudaMallocManaged(&grid_deriv_, DIM * grid_size_ * sizeof(double));
	if(!grid_deriv_) {
	  edm_error("Out of memory!!", "grid.cuh:initialize");	
	}
      }
    }


  
  };
}

#endif //GRID_CUH_
