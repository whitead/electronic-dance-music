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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"GPUassert: \"%s\": %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


namespace EDM{
  
  template< int DIM>
  class DimmedGridGPU : public DimmedGrid<DIM> {
    /** A DIM-dimensional grid for GPU use. Stores on 1D column-ordered array
     **/
  public:
    __host__ DimmedGridGPU(const double* min, 
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
      initialize();
    }


    /**
     * Constructor from file, with interpolation specified
     **/
    DimmedGridGPU(const std::string& input_grid, int b_interpolate): DimmedGrid<DIM> (input_grid, b_interpolate) {
      read(input_grid);
    }

    /**
     * Constructor from grid file
     **/
    DimmedGridGPU(const std::string& input_grid){
      //these are here because of inheritance
      if(grid_ != NULL){
	cudaFree(grid_);
	grid_ = NULL;
      }
      if(grid_deriv_ != NULL){
	cudaFree(grid_deriv_);
	grid_deriv_ = NULL;
      }
      read(input_grid);
    }

    ~DimmedGridGPU() {
      gpuErrchk(cudaDeviceSynchronize());
      if(grid_ != NULL){
	gpuErrchk(cudaFree(grid_));
	grid_ = NULL;//need to do this so DimmedGrid's destructor functions properly
      }
	
      if(grid_deriv_ != NULL){
	gpuErrchk(cudaFree(grid_deriv_));
	grid_deriv_ = NULL;
      }
    }

    virtual void read(const std::string& filename) {
      using namespace std;
      ifstream input;
      size_t i, j;
      input.open(filename.c_str());

      if(!input.is_open()) {      
	cerr << "Cannot open input file \"" << filename <<"\"" <<  endl;
	edm_error("", "grid.h:read");
      }

      // read plumed-style header
      string word;
      input >> word >> word;
      if(word.compare("FORCE") != 0) {
	cerr << "Mangled grid file: " << filename << "No FORCE found" << endl;
	edm_error("", "grid.h:read");
      } else {
	input >> b_derivatives_;
      }
    
      input >> word >> word;
      if(word.compare("NVAR") != 0) {
	cerr << "Mangled grid file: " << filename << " No NVAR found" << endl;
	//edm_error
      } else {
	input >> i;
	if(i != DIM) {
	  cerr << "Dimension of this grid does not match the one found in the file" << endl;
	  edm_error("", "grid.h:read");

	}
      }

      input >> word >> word;
      if(word.compare("TYPE") != 0) {
	cerr << "Mangled grid file: " << filename << " No TYPE found" << endl;
	edm_error("", "grid.h:read");
      } else {
	for(i = 0; i < DIM; i++) {
	  input >> j;
	  if(j != GRID_TYPE) {
	    cerr << "WARNING: Read grid type is the incorrect type" << endl;
	  }
	}
      }

      input >> word >> word;
      if(word.compare("BIN") != 0) {
	cerr << "Mangled grid file: " << filename << " No BIN found" << endl;
	edm_error("", "grid.h:read");
      } else {
	for(i = 0; i < DIM; i++) {
	  input >> grid_number_[i];
	}
      }

      input >> word >> word;
      if(word.compare("MIN") != 0) {
	cerr << "Mangled grid file: " << filename << " No MIN found" << endl;
	edm_error("", "grid.h:read");
      } else {
	for(i = 0; i < DIM; i++) {
	  input >> min_[i];
	}
      }

      input >> word >> word;
      if(word.compare("MAX") != 0) {
	cerr << "Mangled grid file: " << filename << " No MAX found" << endl;
	edm_error("", "grid.h:read");
      } else {
	for(i = 0; i < DIM; i++) {
	  input >> max_[i];
	}
      }

      input >> word >> word;
      if(word.compare("PBC") != 0) {
	cerr << "Mangled grid file: " << filename << " No PBC found" << endl;
	edm_error("", "grid.h:read");
      } else {
	for(i = 0; i < DIM; i++) {
	  input >> b_periodic_[i];
	}
      }

      //now set-up grid number and spacing and preallocate     
      for(i = 0; i < DIM; i++) {
	dx_[i] = (max_[i] - min_[i]) / grid_number_[i];
	if(!b_periodic_[i]) {
	  max_[i] += dx_[i];
	  grid_number_[i] += 1;
	}      
      }
      if(grid_ != NULL) {
	gpuErrchk(cudaFree(grid_));
      }
      if(grid_deriv_ != NULL){
	gpuErrchk(cudaFree(grid_deriv_));
      }
    
      //build arrays
      initialize();
    
      //now we read grid!    
      for(i = 0; i < grid_size_; i++) {
	//skip dimensions
	for(j = 0; j < DIM; j++)
	  input >> word;
	input >> grid_[i];      
	if(b_derivatives_) {
	  for(j = 0; j < DIM; j++) {
	    input >> grid_deriv_[i * DIM + j];
	    grid_deriv_[i * DIM + j] *= -1;
	  }
	}
      }    

      //all done!
      input.close();
    }

    /**
     * Serves the purpose of get_value(), but needs to be callable within and without a GPU kernel
     * Need to have g_grid pointer as an argument because host/device functions get inlined...
     **/
    HOST_DEV double do_get_value( const double* x) const {
      #ifdef __CUDACC__ //device version

      if(b_interpolate_ && b_derivatives_) {//these are pointers
	double temp[DIM];
	return do_get_value_deriv(x, temp);
      }
      size_t index[DIM];
      get_index(x, index);
      printf("do_get_value was called on the GPU with x = %f, and index[0] is now %d\n", x[0], index[0]);
      printf("gpu version of multi2one(index) gives us %d\n", multi2one(index));
      double value = grid_[multi2one(index)];
      printf("and value to be returned is %f\n", value);
      return(value);

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

    HOST_DEV void one2multi(size_t index, size_t result[DIM]) const {
      printf("this is a test. one2multi was called?!?\n");
      int i;
      printf("one2multi was called on GPU with index of %lu\n", index);
      for(i = 0; i < DIM-1; i++) {
	result[i] = index % grid_number_[i];
	index = (index - result[i]) / grid_number_[i];
      }
      result[i] = index;
      printf("made it to the end of one2multi, and index was %lu\n",index);
    }

    HOST_DEV size_t multi2one(const size_t index[DIM]) const {
      size_t result = index[DIM-1];

      size_t i;    
      for(i = DIM - 1; i > 0; i--) {
	result = result * grid_number_[i-1] + index[i-1];
      }
      printf("returning result=%lu\n", result);
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

    double* grid_;
    double* grid_deriv_;

//need to tell compiler where to find these since we have a derived templated class.
    using DimmedGrid<DIM>::grid_size_;
    using DimmedGrid<DIM>::b_derivatives_;
    using DimmedGrid<DIM>::b_interpolate_;
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
      gpuErrchk(cudaMallocManaged(&grid_, grid_size_ * sizeof(double)));
      if(b_derivatives_) {
	gpuErrchk(cudaMallocManaged(&grid_deriv_, DIM * grid_size_ * sizeof(double)));
	if(!grid_deriv_) {
	  edm_error("Out of memory!!", "grid.cuh:initialize");	
	}
      }
      else{
	grid_deriv_ = NULL;
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




}
/*
 * This namespace is used for invoking the GPU functions in the DimmedGridGPU class.
 * Must use a namespace to contain globally-scoped kernel functions.
 * The specific kernels like get_value_kernel are used for unit testing.
 * You MUST cudaMemcpy a DimmedGridGPU object onto the GPU before invoking these kernels.
 */
namespace EDM_Kernels{
  
  using namespace EDM;
  
  /*
   * Kernel wrapper for get_value() on the GPU. Takes in an instance of DimmedGridGPU
   * as well as the address of the coordinate (x) to get the value for, and the target
   * address to store the value, which must be copied to host side if it is to be used there.
   */
  template <int DIM>
  __global__ void get_value_kernel(const double* x, double* target, const DimmedGridGPU<DIM>* g){
    target[0] = g->do_get_value(x);
    printf("get_value_kernel has set target[0] to be %f\n", target[0]);
    return;
  }

  /*
   * Kernel wrapper for multi2one and one2multi testing
   * Takes in target array and temp array to fill as arguments. Validate host-side.
   */
  template <int DIM>
  __global__ void multi2one_kernel(const DimmedGridGPU<DIM>* g, size_t* array, size_t temp[DIM]){
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int j = threadIdx.y + blockIdx.y * blockDim.y;
//    int k = threadIdx.z + blockIdx.z * blockDim.z;
//    if((i < g->grid_number[0] && j < g->grid_number[1]) && k < g->grid_number[2]){
//      array[0] = i;
//      array[1] = j;
//      array[2] = k;
    printf("multi2one_kernel was called!\n");
    g->one2multi(g->multi2one(array), temp);
    printf("made it to the end of multi2one_kernel!\n");
//    }
  }
  
}

#endif //GRID_CUH_
