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

__host__ __device__  int gpu_int_floor(double number) {
  return (int) number < 0.0 ? -ceil(fabs(number)) : floor(number);
}


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

    /**
     * Dispatches cudaFree() on grid_
     **/
    virtual void scrub_grid(){
      cudaFree(grid_);
      grid_ = NULL;
    }

    /**
     * Same principle as scrub_deriv_
     **/
    virtual void scrub_deriv(){
      cudaFree(grid_deriv_);
      grid_deriv_ = NULL;
    }

    /**
     * Serves the purpose of get_value(), but needs to be callable within and without a GPU kernel
     **/
    __host__ __device__ double do_get_value(const double* x) const{
      return(0);
      if(b_interpolate_ && b_derivatives_) {//get "statement is unreachable. Need to fix passing data members"
	double temp[DIM];
	return do_get_value_deriv(x, temp);
      }

      size_t index[DIM];
      get_index(x, index);
      return grid_[multi2one(index)];
    }

    /*
     * Have to re-declare a lot of these so we can use on GPU, 
     * unless we want to mess with the parent classes...
     */
    __host__ __device__ void get_index(const double* x, size_t result[DIM]) const {
      size_t i;
      double xi;
      for(i = 0; i < DIM; i++) {
	xi = x[i];
	if(b_periodic_[i]){
	  xi -= (max_[i] - min_[i]) * gpu_int_floor((xi - min_[i]) / (max_[i] - min_[i]));
	}
	result[i] = (size_t) floor((xi - min_[i]) / dx_[i]);
      }
    }

    __host__ __device__ size_t multi2one(const size_t index[DIM]) const {
      size_t result = index[DIM-1];

      size_t i;    
      for(i = DIM - 1; i > 0; i--) {
	result = result * grid_number_[i-1] + index[i-1];
      }
    
      return result;
    
    }

    __host__ __device__ double do_get_value_deriv(const double* x, double* der) const{
      return 0;
    }
    
    /**
     * Calls the __host__ version of do_get_value() for GPU grids.
     * Can't override get_value() directly due to execution space specifiers.
     **/
    virtual double get_value(const double* x) const{
      if(!(this->in_grid(x))){//doesn't rely on info IN grid, only its dimensions, which are known
	return 0;
      }
      return do_get_value(x);
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

    /**
     * This will actually allocate the arrays and perform any sublcass initialization
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
