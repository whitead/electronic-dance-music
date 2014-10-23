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
                     
inline
double round(double number)
{
  return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

/**
 * This method was adapted from PLUMED 1.3 (Copyright (c) 2008-2011 The PLUMED team.) and is distributed under GPL * v3.
 *
 * The notes below provide a good overview. I've added comments myself to the method throughout
 **/
//-------------------------------------------------------------------------------------------
// Interpolation with a (sort of) cubic spline.
// The function is built as a sum over the nearest neighbours (i.e. 2 in 1d, 4 in 2d, 8 in 3d,...).
// Each neighbour contributes with a polynomial function which is a product of single-dimensional polynomials,
// written as functions of the distance to the neighbour in units of grid spacing
// Each polynomial is proportional to:
// (1-3x^2+2x^3)  + Q (x-2x^2+x^3)
// * its value and derivative in +1 are zero
// * its value in 0 is 1
// * its derivative in 0 is Q
// so, Q is chosen as the desired derivative at the grid point divided by the value at the grid point
// and the final function is multiplied times the value at the grid point.
//
// It works perfectly, except when the tabulated function is zero (there is a special case).
// Maybe one day I will learn the proper way to do splines...
// Giovanni

template<unsigned int DIM> 
double interp(const double* dx, 
	      const double* where, 
	      const double* tabf, 
	      const double* tabder, 
	      const int* stride, 
	      double* der){
// DIM:   dimensionality
// dx:     delta between grid points
// where:  location relative to the floor grid point (always between 0 and dx)
// tabf:   table with function, already pointed at the floor grid point
// tabder: table with POSITIVE gradients (the fastest running index is the dimension index), already pointed at the floor grid point
// stride: strides to the next point on the tabf array.
//         note that, in case of PBC, this stride should corrispond to a backward jump of (N-1) points,
//         where N is the number of points in the domain. 
//         also note that the corrisponding strides for tabder can be obtained multipling times DIM
// der:    in output, the minus gradient.

  int idim, jdim;
  int npoints,ipoint;
  double X;
  double X2;
  double X3; 
  int x0[DIM]; //0 or 1, indicates neighbors relative position on local grid to interpolating point
  double fd[DIM]; // the unscaled interpolated derivative contribution from a neighbor
  double C[DIM]; // The value of the polynomials
  double D[DIM]; // The derivative of the polynomials
  int  tmp,shift;
  double f; //the final value of the interpolated function
  double ff; // the unscaled interpolated contribution from a neighbor
  double qq; // the neighbor derivative divided by its value

  //Get the number of neighbors for a given dimension
  npoints = 1; 
  for(idim = 0;idim < DIM; idim++) 
    npoints *= 2; // npoints=2**DIM
  
  // reset
  f = 0;
  for(idim = 0;idim < DIM; idim++) 
    der[idim] = 0;
  
  // looping over neighbour points:
  for(ipoint = 0; ipoint < npoints; ipoint++){
    
    // find the local grid offset of neighbour point (x0) [0 or 1] and its corresponding change in collapsed index in order to use the given potential and forces
    tmp = ipoint;
    shift = 0;
    for(idim = 0;idim < DIM; idim++){
      x0[idim] = tmp % 2; tmp /= 2; //this defines the single current neighbor we're considering 
      shift += stride[idim] * x0[idim]; //the shift in the collapsed index for the current neighbor
    }

    // reset contribution
    ff = 1.0;

    //the neighbor has n-dimensions, so we make the polynomial n-dimensional
    for(idim = 0;idim < DIM; idim++){
      X = fabs(where[idim] / dx[idim] - x0[idim]); //switch from local spatial coordinates to local rescaled 
      X2 = X * X;
      X3 = X2 * X;
      if(fabs(tabf[shift]) < 0.0000001) 
	qq = 0.0; //special case of 0/0
      else 
	qq = -tabder[shift * DIM + idim] / tabf[shift];
      
      //The sign change is in case we want a backwards or fowrards derivative
      //dx comes back in via chain rule
      C[idim] = (1 - 3 * X2 + 2 * X3) - (x0[idim] ? -1 : 1) * qq * (X - 2 * X2 + X3) * dx[idim];
      D[idim] = ( -6 * X + 6 * X2) - (x0[idim] ? -1 : 1) * qq * (1 - 4 * X + 3 * X2) * dx[idim]; // d / dX
      D[idim]  *= (x0[idim] ? -1 : 1)/dx[idim]; // chain rule (to where)
      ff  *= C[idim]; // we are multiplying the polynomials (same as below)
    }

    for(idim = 0; idim < DIM; idim++) {
      fd[idim] = D[idim];
      for(jdim = 0; jdim < DIM; jdim++) 
	if(jdim != idim) 
	  fd[idim]  *= C[jdim];
    }

    //add the contribution from this neighbor
    f += tabf[shift]*ff;
    for(idim = 0; idim < DIM; idim++) 
      der[idim] += tabf[shift] * fd[idim];
  }
  return f;
};


class Grid {


 public:  
  virtual double get_value(const double* x) const = 0;
  /**
   * Get value and put derivatives into "der"
   **/
  virtual double get_value_deriv(const double* x, double* der) const = 0;
  /**
   * Write the grid to the given file
   **/
  virtual void write(const std::string& filename) const = 0;
  virtual void read(const std::string& filename) = 0;
  virtual void set_interpolation(int b_interpolate) = 0;
  virtual double* get_grid() = 0;
  virtual size_t get_grid_size() const = 0;

};

template<unsigned int DIM>
class DimmedGrid : public Grid {
  /** A DIM-dimensional grid for storing things. Stores on 1D column-ordered array
   *
   **/
 public:
 DimmedGrid(const double* min, 
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

 DimmedGrid(const std::string& input_grid): b_derivatives_(0), b_interpolate_(0), grid_(NULL), grid_deriv_(NULL) {
    read(input_grid);
  }

  /** 
   * Clone constructor
   **/
 DimmedGrid(const DimmedGrid<DIM>& other) : b_derivatives_(other.b_derivatives_), b_interpolate_(other.b_interpolate_), grid_(NULL), grid_deriv_(NULL) {
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

  ~DimmedGrid() {
    if(grid_ != NULL)
      free(grid_);
    if(grid_deriv_ != NULL)
      free(grid_deriv_);
    
  }
  
  /**
   * Go from a point to an array of indices
   **/ 
  void get_index(const double* x, size_t result[DIM]) const {
    size_t i;
    double xi;
    for(i = 0; i < DIM; i++) {
      xi = x[i];
      if(b_periodic_[i])
	xi -= (max_[i] - min_[i]) * int_floor((xi - min_[i]) / (max_[i] - min_[i]));
      result[i] = (int) floor((xi - min_[i]) / dx_[i]);
    }
  }

  /**
   * Go from an array of indices to a single index
   **/
  size_t multi2one(const size_t index[DIM]) const {
    size_t result = index[DIM-1];

    size_t i;    
    for(i = DIM - 1; i > 0; i--) {
      result = result * grid_number_[i-1] + index[i-1];
    }
    
    return result;
    
  }

  /** 
   * Go from single index to array
   **/
  void one2multi(size_t index, size_t result[DIM]) const {
    size_t i;

    for(i = 0; i < DIM-1; i++) {
      result[i] = index % grid_number_[i];
      index = (index - result[i]) / grid_number_[i];
    }
    result[i] = index; 
  }

  /**
   * Get the value of the grid at x
   **/ 
  double get_value(const double* x) const{

    check_point(x);
      
    if(b_interpolate_) {
      double temp[DIM];
      return get_value_deriv(x, temp);
    }

    size_t index[DIM];
    get_index(x, index);
    return grid_[multi2one(index)];
  }

  /**
   * Get value and derivatives
   **/ 
  double get_value_deriv(const double* x, double* der) const {
    
    if(grid_deriv_ == NULL) {
      std::cerr << "This grid has no derivatives" << std::endl;
      //error
      return get_value(x);
    }

    double value;
    size_t index[DIM];
    size_t index1;
    size_t i;
    
    //checks
    check_point(x);

    get_index(x, index);
    index1 = multi2one(index);

    if(b_interpolate_) {
      
      double where[DIM]; //local position (local meaning relative to neighbors
      int stride[DIM]; //the indexing stride, which also accounts for periodicity
      double wrapped_x;

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
	if(b_periodic_[i] && index[i] == grid_number_[i] - 1)
	  stride[i] *= (1 - grid_number_[i]);
      }
      
      value = interp<DIM>(dx_, where, &grid_[index1], &grid_deriv_[index1 * DIM], stride, der);
      
    } else {
      for(i = 0; i < DIM; i++) {
	der[i] = grid_deriv_[index1 * DIM + i];
      }
      value = grid_[index1];
    }

    return value;
  }
   
  void write(const std::string& filename) const {
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

  void set_interpolation(int b_interpolate) {
    b_interpolate_ = b_interpolate;
  }
 
  double* get_grid() {
    return grid_;
  }

  size_t get_grid_size() const{
    return grid_size_;
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
  /**
   * Check if a point is in bounds
   **/
  int check_point(const double x[DIM]) const {
    size_t i;
    for(i = 0; i < DIM; i++) {
      if((x[i] < min_[i] || x[i] > max_[i]) && !b_periodic_[i]){
	std::cerr << "Bad grid value " << x[i] << " in dimension " << i << std::endl;
	//error
	return 0;
      }
    }
    return 1;
  }
  

  /** This will actually allocate the arrays and perform any sublcass initialization
   *
   **/
  void initialize() {
    size_t i;
    grid_size_ = 1;
    for(i = 0; i < DIM; i++)
      grid_size_ *= grid_number_[i];
    grid_ = (double *) calloc(DIM * grid_size_, sizeof(double));
    if(b_derivatives_)
      grid_deriv_ = (double *) calloc(DIM * DIM * grid_size_, sizeof(double));
  }

  
};

Grid* make_grid(unsigned int dim, 
		const double* min, 
		const double* max, 
		const double* bin_spacing, 
		const int* b_periodic, 
		int b_derivatives, 
		int b_interpolate) {
  switch(dim) {
  case 1:
    return new DimmedGrid<1>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  case 2:
    return new DimmedGrid<2>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  case 3:
    return new DimmedGrid<3>(min, max, bin_spacing, b_periodic, b_derivatives, b_interpolate);
  }

  return NULL;
}

Grid* read_grid(unsigned int dim, const std::string& filename) {
  switch(dim) {
  case 1:
    return new DimmedGrid<1>(filename);
  case 2:
    return new DimmedGrid<2>(filename);
  case 3:
    return new DimmedGrid<3>(filename);
  }
  return NULL;
}


#endif //GRID_H_
