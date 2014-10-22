#ifndef GRID_H_
#define GRID_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>



template<unsigned int DIM>
class Grid {
  /** A DIM-dimensional grid for storing things. Stores on 1D column-ordered array
   *
   **/
 public:
Grid(double* min, double* max, double* bin_spacing, int* b_periodic, int b_derivatives) : b_derivatives_(b_derivatives) {

    unsigned int i;

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
  }

  /** This will actually allocate the arrays and perform any sublcass initialization
   *
   **/
  void initialize() {
    unsigned int i;
    grid_size_ = 1;
    for(i = 0; i < DIM; i++)
      grid_size_ *= grid_number_[i];
    grid_ = (double *) malloc(sizeof(double) * DIM * grid_size_);
    if(b_derivatives_)
      grid_deriv_ = (double *) malloc(sizeof(double) * DIM * grid_size_ * 3);
  }
  
  /**
   * Go from a point to an array of indices
   **/ 
  size_t* get_index(double* x, size_t* result) {
    unsigned int i;
    for(i = 0; i < DIM; i++) {
      result[i] = (int) floor((x[i] - min_[i]) / dx_[i]);
    }
    return result;
  }

  /**
   * Go from an array of indices to a single index
   **/
  int multi2one(size_t index[DIM]) {
    size_t result = index[DIM-1];

    unsigned int i;    
    for(i = DIM - 1; i > 0; i--) {
      result = result * grid_number_[i-1] + index[i-1];
    }
    
    return result;
    
  }

  /**
   * Get the value of the grid at x
   **/ 
  double get_value(double* x) {
    size_t index[DIM];
    get_index(x, index);
    return grid_[multi2one(index)];
  }
   
  void write(std::string& filename) {
    //write grid to a file
  }

  void read(std::string& filename) {
    //read grid from a file
  }
  
  double dx_[DIM];//grid spacing
  double min_[DIM];//grid minimum
  double max_[DIM];//maximum
  int grid_number_[DIM];//number of points on grid
  long grid_size_; //total grid size
  int b_periodic_[DIM];//if a dimension is periodic
  int b_derivatives_;//if derivatives are going to be used
  double* grid_;//the grid values
  double* grid_deriv_;//derivatives  
  
  
};

#endif //GRID_H_
