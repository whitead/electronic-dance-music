#ifndef GRID_H_
#define GRID_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>


inline
int int_floor(double number) {
  return (int) number < 0.0 ? -ceil(fabs(number)) : floor(number);                                                                   
}                                                                                                                   

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
  size_t* get_index(double* x, size_t result[DIM]) {
    unsigned int i;
    for(i = 0; i < DIM; i++) {
      if(b_periodic_[i])
	x[i] -= (max_[i] - min_[i]) * int_floor((x[i] - min_[i]) / (max_[i] - min_[i]));
      result[i] = (int) floor((x[i] - min_[i]) / dx_[i]);
    }
    return result;
  }

  /**
   * Go from an array of indices to a single index
   **/
  size_t multi2one(size_t index[DIM]) {
    size_t result = index[DIM-1];

    unsigned int i;    
    for(i = DIM - 1; i > 0; i--) {
      result = result * grid_number_[i-1] + index[i-1];
    }
    
    return result;
    
  }

  /** 
   * Go from single index to array
   **/
  void one2multi(size_t index, size_t result[DIM]) {
    unsigned int i;

    for(i = 0; i < DIM-1; i++) {
      result[i] = index % grid_number_[i];
      index = (index - result[i]) / grid_number_[i];
    }
    result[i] = index; 
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
    using namespace std;
    ofstream output;
    size_t i, j;
    output.open(filename);

    // print plumed-style header
    output << "#! FORCE " << b_derivatives_ << endl;
    output << "#! NVAR " << DIM << endl;
    output << "#! TYPE " << 32 << endl;

    output << "#! BIN ";
    for(i = 0; i < DIM; i++)
      output << b_periodic_[i] ? grid_number_[i] - 1 : grid_number_[i];
    output << endl;

    output << "#! MIN ";
    for(i = 0; i < DIM; i++)
      output << min_[i];
    output << endl;

    output << "#! MAX ";
    for(i = 0; i < DIM; i++)
      output << b_periodic_[i] ? max_[i] : max_[i] - dx_[i];
    output << endl;

    output << "#! PBC ";
    for(i = 0; i < DIM; i++)
      output << b_periodic_[i];
    output << endl;


    //print out grid
    double point;
    size_t temp[DIM];
    for(i = 0; i < grid_size_; i++) {
      one2multi(i, temp);
      for(j = 0; j < DIM; j++) {
	output << setw(8) << (min_[j] + dx_[j] * temp[j]) << " ";
      }
      output << setw(8) << grid_[i] << " ";
      if(b_derivatives_) {
	for(j = 0; j < 3; j++) {
	  output << setw(8) << grid_deriv_[i*3 + j] << " ";
	}
      }
      output << endl;
      if(temp[0] == grid_number[0] - 1)
	output << endl;
    }

  }

  void read(std::string& filename) {
    using namespace std;
    ifstream input;
    size_t i, j;
    input.open(filename);

    // read plumed-style header
    std::string word;
    input >> word >> word;
    if(word.compare("FORCE") != 0)
      fprintf(stderr, "Mangled grid file\n");
    
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
