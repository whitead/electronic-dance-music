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

      size_t i;
      //need to set these explicitly due to some inheritance issue I don't understand--Rainier
      b_derivatives_ = b_derivatives;
      b_interpolate_ = b_interpolate;

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

    /**
     * Constructor from file, with interpolation specified
     **/
    DimmedGridGPU(const std::string& input_grid, int b_interpolate):DimmedGrid<DIM> (input_grid, b_interpolate){
      read(input_grid);
    }

    /**
     * Constructor from grid file
     **/
    DimmedGridGPU(const std::string& input_grid):DimmedGrid<DIM> (input_grid){
      read(input_grid);
    }

    ~DimmedGridGPU() {
      if(grid_ != NULL)
	cudaFree(grid_);
      if(grid_deriv_ != NULL)
	cudaFree(grid_deriv_);
    }

    double get_value(const double* x) const{
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

    void read(const std::string& filename){
      
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
	cudaFree(grid_);
	grid_ = NULL;
      }
      if(grid_deriv_ != NULL){
	cudaFree(grid_deriv_);
	grid_deriv_ = NULL;
      }
    
      //build arrays
      this->initialize();
    
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
    void initialize() {//this cudamallocs our device grid_ & grid_deriv_ pointers
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
