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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true, bool print=true)
{
  if (code != cudaSuccess) 
  {
    if(print){
      fprintf(stderr,"GPUassert: \"%s\": %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
    }
    if (abort) exit(code);
  }
}

#define gpuErrchkNoQuit(ans) { gpuAssert((ans), __FILE__, __LINE__, false, false);}


namespace EDM{

  
  
  template< int DIM>
  class DimmedGridGPU : public DimmedGrid<DIM> {
    /** A DIM-dimensional grid for GPU use. Stores on 1D column-ordered array
     **/
  public:
    __host__ DimmedGridGPU(const edm_data_t* min, 
			   const edm_data_t* max, 
			   const edm_data_t* bin_spacing, 
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
     * Default constructor

     DimmedGridGPU():b_derivatives_(0), b_interpolate_(1), grid_(NULL), grid_deriv_(NULL){}
    **/


    /**
     * Constructor from file, with interpolation specified
     **/
    DimmedGridGPU(const std::string& input_grid, int b_interpolate): DimmedGrid<DIM> (input_grid, b_interpolate) {
      read(input_grid);
    }

    /**
     * Constructor from grid file
     **/
    DimmedGridGPU(const std::string& input_grid): DimmedGrid<DIM>(input_grid){
      //these are here because of inheritance
      gpuErrchk(cudaDeviceSynchronize());
      if(grid_ != NULL){
#ifdef __CUDACC__
	cudaPointerAttributes* attributes = new cudaPointerAttributes;
	try{
	  gpuErrchkNoQuit(cudaPointerGetAttributes(attributes, grid_));
	}
	catch(const std::exception&){
	  free(grid_);
	}
#else
	free(grid_);
#endif //CUDACC
	grid_ = NULL;
      }
      if(grid_deriv_ != NULL){
#ifdef __CUDACC__
	cudaPointerAttributes* attributes = new cudaPointerAttributes;
	try{
	  gpuErrchkNoQuit(cudaPointerGetAttributes(attributes, grid_deriv_));
	}
	catch(const std::exception&){
	  free(grid_deriv_);
	}
#else
	free(grid_deriv_);
#endif //CUDACC
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



    /**
     * Serves the purpose of get_value(), but needs to be callable within and without a GPU kernel
     * Need to have g_grid pointer as an argument because host/device functions get inlined...
     **/
    HOST_DEV edm_data_t do_get_value( const edm_data_t* x) const {
      if(!in_grid(x)){
	return(0);
      }
#ifdef __CUDACC__ //device version

      if(d_b_interpolate_[0] && d_b_derivatives_[0]) {//these are pointers
	edm_data_t temp[DIM];
	return do_get_value_deriv(x, temp);
      }
      size_t index[DIM];
      get_index(x, index);//looks like this is the culprit. Rounding error?
      return(grid_[multi2one(index)]);

#else//host version
      return get_value(x);
#endif// CUDACC
    }

    /*
     * Have to re-declare a lot of these so we can use on GPU, 
     * unless we want to mess with the parent classes...
     */
    HOST_DEV void get_index(const edm_data_t* x, size_t result[DIM]) const {
      size_t i;
      edm_data_t xi;
      for(i = 0; i < DIM; i++) {
	xi = x[i];
	if(b_periodic_[i]){
	  xi -= (max_[i] - min_[i]) * int_floor((xi - min_[i]) / (max_[i] - min_[i]));
	}
	result[i] = (size_t) floor((xi - min_[i]) / dx_[i]);
      }
    }

    //takes in 1D index, modifies result to be full of the DIM-D coordinates
    HOST_DEV void one2multi(size_t index, size_t result[DIM]) const {
      int i;
      for(i = 0; i < DIM-1; i++) {
	result[i] = index % d_grid_number_[i];
	index = (index - result[i]) / d_grid_number_[i];
      }
      result[i] = index;
    }

    HOST_DEV size_t multi2one(const size_t index[DIM]) const {
      size_t result = index[DIM-1];

      size_t i;    
      for(i = DIM - 1; i > 0; i--) {
	result = result * d_grid_number_[i-1] + index[i-1];
      }
      return result;
    
    }

    HOST_DEV bool in_grid(const edm_data_t x[DIM]) const {
      size_t i;
      for(i = 0; i < DIM; i++) {
	if(!b_periodic_[i] && (x[i] < min_[i] || x[i] > (max_[i])) ){
	  return false;
	}
      }
      return true;
    }


    HOST_DEV edm_data_t do_get_value_deriv(const edm_data_t* x, edm_data_t* der) const{
      // if(d_b_derivatives_[0]){
      // 	printf("I GUESS DERIVATIVES WAS TRUE ON GPU\n");
      // }
      size_t i;
      if(!in_grid(x)) {
	for(i = 0; i < DIM; i++)
	  der[i] = 0;
	return 0;
      }
      edm_data_t value;
      size_t index1;
      size_t index[DIM];
      get_index(x, index);
      index1 = multi2one(index);
      if(b_interpolate_) {
      
	edm_data_t where[DIM]; //local position (local meaning relative to neighbors)
	int stride[DIM]; //the indexing stride, which also accounts for periodicity
	edm_data_t wrapped_x;
      
	stride[0] = 1; //dim 0 is fastest
	for(i = 1; i < DIM; i++)
	  stride[i] = stride[i - 1] * grid_number_[i - 1];

	for(i = 0; i < DIM; i++) {
	  //wrap x, if needed
	  wrapped_x = x[i];
	  if(b_periodic_[i])
	    wrapped_x -= (max_[i] - min_[i]) * int_floor((wrapped_x - min_[i]) / (max_[i] - min_[i]));
	  //get position relative to neighbors
	  where[i] = wrapped_x - min_[i] - index[i] * dx_[i];
	  //treat possible stride wrap
	  if(b_periodic_[i] && index[i] == grid_number_[i] - 1){
//	  printf("adjusting for being at the right edge\n");
	    stride[i] *= (1 - grid_number_[i]);
	  }
	  
	}
	
	value = interp<DIM>(dx_, where, &grid_[index1], &grid_deriv_[index1 * DIM], stride, der);
      
      } else {
	for(i = 0; i < DIM; i++) {
	  der[i] = grid_deriv_[index1 * DIM + i];
	}
	value = grid_[index1];
      }

      return value;


//      return 3.1415926535897932;
    }
    /*
     * Must override the read function or else include CUDA code in the base
     * grid.h file... =/
     */
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
      if(grid_ != NULL){
#ifdef __CUDACC__
	cudaPointerAttributes* attributes = new cudaPointerAttributes;
	try{
	  gpuErrchkNoQuit(cudaPointerGetAttributes(attributes, grid_));
	}
	catch(const std::exception&){
	  free(grid_);
	}
#else
	free(grid_);
#endif //CUDACC
	grid_ = NULL;
      }
      if(grid_deriv_ != NULL){
#ifdef __CUDACC__
	cudaPointerAttributes* attributes = new cudaPointerAttributes;
	try{
	  gpuErrchkNoQuit(cudaPointerGetAttributes(attributes, grid_deriv_));
	}
	catch(const std::exception&){
	  free(grid_deriv_);
	}
#else
	free(grid_deriv_);
#endif //CUDACC
	grid_deriv_ = NULL;
      }
      // if(grid_ != NULL) {
      //   free(grid_);
      // }
      // if(grid_deriv_ != NULL){
      //   free(grid_deriv_);
      // }
    
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
     * Called by the __host__ version of do_get_value() for GPU grids.
     **/
    virtual edm_data_t get_value(const edm_data_t* x) const{
      if(!(this->in_grid(x))){
	return 0;
      }
      if(b_interpolate_ && b_derivatives_) {
	edm_data_t temp[DIM];
	return this->get_value_deriv(x, temp);
      }

      size_t index[DIM];
      get_index(x, index);
      return grid_[multi2one(index)];
    }

//    edm_data_t* grid_;
//    edm_data_t* grid_deriv_;
    int* d_grid_number_;//device grid number arr
    int* d_b_derivatives_;//device 'bool' for whether we're using derivatives
    int* d_b_interpolate_;//device 'bool' for whether we're interpolating

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
    using DimmedGrid<DIM>::read;

  private:  

    /**
     * This will actually allocate the arrays and perform any sublcass initialization
     **/
    virtual void initialize() {//this cudamallocs our device grid_ & grid_deriv_ pointers
      size_t i;
      grid_size_ = 1;
      gpuErrchk(cudaMallocManaged(&d_grid_number_, DIM*sizeof(int)));
      gpuErrchk(cudaMallocManaged(&d_b_derivatives_, sizeof(int)));
      gpuErrchk(cudaMallocManaged(&d_b_interpolate_, sizeof(int)));
      d_b_derivatives_[0] = b_derivatives_;//need these for device do_get_value function
      d_b_interpolate_[0] = b_interpolate_;//need these for device do_get_value function
      
      for(i = 0; i < DIM; i++){
	grid_size_ *= grid_number_[i];
	d_grid_number_[i] = grid_number_[i];
      }
      gpuErrchk(cudaMallocManaged(&grid_, grid_size_ * sizeof(edm_data_t)));
      if(b_derivatives_) {
	gpuErrchk(cudaMallocManaged(&grid_deriv_, DIM * grid_size_ * sizeof(edm_data_t)));
	if(!grid_deriv_) {
	  edm_error("Out of memory!!", "grid.cuh:initialize");	
	}
      }
      else{
	grid_deriv_ = NULL;
      }
      gpuErrchk(cudaDeviceSynchronize());
    }


    
  };
}
/*
 * This namespace is used for invoking the GPU functions in the DimmedGridGPU class.
 * Must use a namespace to contain globally-scoped kernel functions.
 * The specific kernels like get_value_kernel are used for unit testing.
 * MUST cudaMemcpy a DimmedGridGPU object onto the GPU before invoking these kernels.
 */
namespace EDM_Kernels{
  
  using namespace EDM;
  
  /*
   * Kernel wrapper for get_value() on the GPU. Takes in an instance of DimmedGridGPU
   * as well as the address of the coordinate (x) to get the value for, and the target
   * address to store the value, which must be copied to host side if it is to be used there.
   */
  template <int DIM>
  __global__ void get_value_kernel(const edm_data_t* x, edm_data_t* target,
				   const DimmedGridGPU<DIM>* g){
    target[0] = g->do_get_value(x);
    return;
  }

  template <int DIM>
  __global__ void get_value_deriv_kernel(const edm_data_t* x, edm_data_t* der, edm_data_t* target,
					 const DimmedGridGPU<DIM>* g){
    target[0] = g->do_get_value_deriv(x, der);
    return;
  }

  /*
   * Kernel wrapper for multi2one and one2multi testing
   * Takes in target array and temp array to fill as arguments. Validate host-side.
   */
  template <int DIM>
  __global__ void multi2one_kernel(size_t* array, size_t* temp, const DimmedGridGPU<DIM>* g){
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int j = threadIdx.y + blockIdx.y * blockDim.y;
//    int k = threadIdx.z + blockIdx.z * blockDim.z;
//    if((i < g->grid_number[0] && j < g->grid_number[1]) && k < g->grid_number[2]){
//      array[0] = i;
//      array[1] = j;
//      array[2] = k;

    g->one2multi(g->multi2one(array), temp);
//    }
  }
  
}

#endif //GRID_CUH_
