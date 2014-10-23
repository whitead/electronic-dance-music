#ifndef GRID_H_
#define GRID_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#define GRID_TYPE 32
 

inline
int int_floor(double number) {
  return (int) number < 0.0 ? -ceil(fabs(number)) : floor(number);                                                                   
}                                                                                                                   

class Grid {


 public:  
 Grid(int b_derivatives, double* grid, double* grid_deriv) : b_derivatives_(b_derivatives), 
    grid_(grid), grid_deriv_(grid_deriv) {
    //empty
  }
  virtual double get_value(double* x) = 0;
  virtual void write(const std::string& filename) = 0;
  virtual void read(const std::string& filename) = 0;
  virtual void initialize() = 0;

  size_t grid_size_;//total size of grid
  int b_derivatives_;//if derivatives are going to be used
  double* grid_;//the grid values
  double* grid_deriv_;//derivatives  
};

template<unsigned int DIM>
class DimmedGrid : public Grid {
  /** A DIM-dimensional grid for storing things. Stores on 1D column-ordered array
   *
   **/
 public:
 DimmedGrid(double* min, double* max, double* bin_spacing, int* b_periodic, int b_derivatives) : Grid(b_derivatives, NULL, NULL) {

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

 DimmedGrid(const std::string& input_grid): Grid(0, NULL, NULL) {
    read(input_grid);
  }

  /** 
   * Clone constructor
   **/
 DimmedGrid(DimmedGrid<DIM>& other) : Grid(other.b_derivatives_, NULL, NULL) {
    size_t i,j;
    for(i = 0; i < DIM; i++) {
      dx_[i] = other.dx_[i];
      min_[i] = other.min_[i];
      max_[i] = other.max_[i];
      grid_number_[i] = other.grid_number_[i];
      b_periodic_[i] = other.b_periodic_[i];
    }
    
    initialize();
    for(i = 0; i < grid_size_; i++) {
      grid_[i] = other.grid_[i];
      if(b_derivatives_) {
	for(j = 0; j < DIM; j++)
	  grid_deriv_[i * DIM + j] = other.grid_deriv_[i * DIM + j];
      }
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
      grid_deriv_ = (double *) malloc(sizeof(double) * DIM * DIM * grid_size_);
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
   
  void write(const std::string& filename) {
    using namespace std;
    ofstream output;
    size_t i, j;
    output.open(filename);

    // print plumed-style header
    output << "#! FORCE " << b_derivatives_ << endl;
    output << "#! NVAR " << DIM << endl;

    output << "#! TYPE ";
    for(i = 0; i < DIM; i++)
      output << GRID_TYPE << " ";
    output << endl;

    output << "#! BIN ";
    for(i = 0; i < DIM; i++)
      output << (b_periodic_[i] ? grid_number_[i] : grid_number_[i] - 1) << " ";
    output << endl;

    output << "#! MIN ";
    for(i = 0; i < DIM; i++)
      output << min_[i] << " ";
    output << endl;

    output << "#! MAX ";
    for(i = 0; i < DIM; i++)
      output << (b_periodic_[i] ? max_[i] : max_[i] - dx_[i]) << " ";
    output << endl;

    output << "#! PBC ";
    for(i = 0; i < DIM; i++)
      output << b_periodic_[i] << " ";
    output << endl;


    //print out grid
    double point;
    size_t temp[DIM];
    for(i = 0; i < grid_size_; i++) {
      one2multi(i, temp);
      for(j = 0; j < DIM; j++) {
	output << setw(8) << left << setfill('0') << (min_[j] + dx_[j] * temp[j]) << " ";
      }
      output << setw(8) << left << setfill('0') << grid_[i] << " ";
      if(b_derivatives_) {
	for(j = 0; j < DIM; j++) {
	  output << setw(8) << left << setfill('0')<<  grid_deriv_[i*DIM + j] << " ";
	}
      }
      output << endl;
      if(temp[0] == grid_number_[0] - 1)
	output << endl;
    }

    output.close();    
  }
  
  void read(const std::string& filename) {
    using namespace std;
    ifstream input;
    size_t i, j;
    input.open(filename);

    if(!input.is_open()) {      
      cerr << "Cannot open input file " << filename << endl;
      //error
    }

    // read plumed-style header
    string word;
    input >> word >> word;
    if(word.compare("FORCE") != 0) {
      cerr << "Mangled grid file: " << filename << "No FORCE found" << endl;
      //error
    } else {
      input >> b_derivatives_;
    }
    
    input >> word >> word;
    if(word.compare("NVAR") != 0) {
      cerr << "Mangled grid file: " << filename << " No NVAR found" << endl;
      //error
    } else {
      input >> i;
      if(i != DIM) {
	cerr << "Dimension of this grid does not match the one found in the file" << endl;
	//error
      }
    }

    input >> word >> word;
    if(word.compare("TYPE") != 0) {
      cerr << "Mangled grid file: " << filename << " No TYPE found" << endl;
      //error
    } else {
      for(i = 0; i < DIM; i++) {
	input >> j;
	if(j != GRID_TYPE) {
	  cerr << "This grid is the incorrect type" << endl;
	  //error
	}
      }
    }

    input >> word >> word;
    if(word.compare("BIN") != 0) {
      cerr << "Mangled grid file: " << filename << " No BIN found" << endl;
    } else {
      for(i = 0; i < DIM; i++) {
	input >> grid_number_[i];
      }
    }

    input >> word >> word;
    if(word.compare("MIN") != 0) {
      cerr << "Mangled grid file: " << filename << " No MIN found" << endl;
    } else {
      for(i = 0; i < DIM; i++) {
	input >> min_[i];
      }
    }

    input >> word >> word;
    if(word.compare("MAX") != 0) {
      cerr << "Mangled grid file: " << filename << " No MAX found" << endl;
    } else {
      for(i = 0; i < DIM; i++) {
	input >> max_[i];
      }
    }

    input >> word >> word;
    if(word.compare("PBC") != 0) {
      cerr << "Mangled grid file: " << filename << " No PBC found" << endl;
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
    if(grid_ != NULL)
      free(grid_);
    if(grid_deriv_ != NULL)
      free(grid_deriv_);
    
    //build arrays
    initialize();
    
    //now we read grid!    
    size_t temp[DIM];
    for(i = 0; i < grid_size_; i++) {
      //skip dimensions
      for(j = 0; j < DIM; j++)
	input >> word;
      input >> grid_[i];
      if(b_derivatives_) {
	for(j = 0; j < DIM; j++) {
	  input >> grid_deriv_[i * DIM + j];
	}
      }
    }    

    //all done!
    input.close();
  }
  
  double dx_[DIM];//grid spacing
  double min_[DIM];//grid minimum
  double max_[DIM];//maximum
  int grid_number_[DIM];//number of points on grid
  int b_periodic_[DIM];//if a dimension is periodic
  
};

#endif //GRID_H_
