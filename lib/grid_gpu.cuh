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

HOST_DEV  int gpu_int_floor(double number) {
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
      cudaMalloc((void**)&d_b_interpolate_, sizeof(int));
      cudaMemcpy(d_b_interpolate_, &b_interpolate, sizeof(int), cudaMemcpyHostToDevice);
      cudaMalloc((void**)&d_b_derivatives_, sizeof(int));
      cudaMemcpy(d_b_derivatives_, &b_derivatives, sizeof(int), cudaMemcpyHostToDevice);

    }


    /**
     * Constructor from file, with interpolation specified
     **/
    DimmedGridGPU(const std::string& input_grid, int b_interpolate): DimmedGrid<DIM> (input_grid, b_interpolate) {
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
      
      cudaDeviceReset();
    }


    /**
     * Serves the purpose of get_value(), but needs to be callable within and without a GPU kernel
     **/
    HOST_DEV double do_get_value(const double* x) const{
      #ifdef __CUDACC__ //device version
      if(d_b_interpolate_[0] && d_b_derivatives_[0]) {//these are pointers
	double temp[DIM];
	return do_get_value_deriv(x, temp);
      }
      
      size_t index[DIM];
      get_index(x, index);
      return grid_[multi2one(index)];

      #else//host version
      return get_value(x);
      #endif// CUDACC
    }

    /*
     * Have to re-declare a lot of these so we can use on GPU, 
     * unless we want to mess with the parent classes...
     */
    HOST_DEV void get_index(const double* x, size_t result[DIM]) const {
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

    HOST_DEV size_t multi2one(const size_t index[DIM]) const {
      size_t result = index[DIM-1];

      size_t i;    
      for(i = DIM - 1; i > 0; i--) {
	result = result * grid_number_[i-1] + index[i-1];
      }
    
      return result;
    
    }

    HOST_DEV double do_get_value_deriv(const double* x, double* der) const{
      return 1.0;
    }
    
    /**
     * Calls the __host__ version of do_get_value() for GPU grids.
     * Can't override get_value() directly due to execution space specifiers.
     **/
    virtual double get_value(const double* x) const{
      if(!(this->in_grid(x))){
	return 0;
      }
            
      if(b_interpolate_ && b_derivatives_) {
	double temp[DIM];
	return this->get_value_deriv(x, temp);
      }

      size_t index[DIM];
      get_index(x, index);
      return grid_[multi2one(index)];
    }

    int* d_b_interpolate_;
    int* d_b_derivatives_;


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

  /*
   *    HOST_DEV double do_get_value(const double* x) const{
 *     #ifdef __CUDACC__
 *     if(d_b_interpolate_[0] && d_b_derivatives_[0]) {//these are pointers
 *	double temp[DIM];
 *	return do_get_value_deriv(x, temp);
 *    }
 */

  //global functions need global scope...
  template <int DIM>
  __global__ void get_value_kernel(const double* x, double* target, DimmedGridGPU<DIM> g){
    printf("Hello from the GPU land! g.d_b_derivatives_[0] is %d\n", g.d_b_derivatives_[0]);
      target[0] = g.do_get_value(x);
  }


}

#endif //GRID_CUH_
