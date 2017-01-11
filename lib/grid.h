#ifndef GRID_H_
#define GRID_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "mpi.h"

#include "edm.h"
#ifndef GRID_TYPE
#define GRID_TYPE 32
#endif //GRID_TYPE

inline
 int int_floor(double number) {
  return (int) number < 0.0 ? -ceil(fabs(number)) : floor(number);                                                                   
}     
                     
namespace EDM{

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

template<int DIM> 
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
//         note that, in case of PBC, this stride should correspond to a backward jump of (N-1) points,
//         where N is the number of points in the domain. 
//         also note that the corrisponding strides for tabder can be obtained multipling times DIM
// der:    in output, the POSITIVE gradient.

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
//    printf("made it to ipoint = %i\n", ipoint);
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
	qq = -tabder[shift * DIM + idim] / tabf[shift];//[rainier] invalid read from this line
      
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
  virtual  double get_value(const double* x) const = 0;
  /**
   * Adds the given value to the grid and then returns how much was
   * actually added. The discrepancy can arise due to boundaries
   * and/or kernels.
   */
  virtual  double add_value(const double* x0, double value) = 0;
  virtual ~Grid() {};
  /**
   * Get value and put derivatives into "der"
   **/
  virtual  double get_value_deriv(const double* x, double* der) const = 0;
  /**
   * Write the grid to the given file
   **/
  virtual void write(const std::string& filename) const = 0;
  virtual void multi_write(const std::string& filename, 
			   const double* box_low, 
			   const double* box_high, 
			   const int* b_periodic,
			   int b_lammps_format) const = 0;
  virtual void read(const std::string& filename) = 0;
  virtual void set_interpolation(int b_interpolate) = 0;
  virtual double* get_grid() = 0;
  virtual const double* get_dx() const = 0;
  virtual const double* get_max() const = 0;
  virtual const double* get_min() const = 0;
  virtual double max_value() const = 0;
  virtual double min_value() const = 0;
  virtual void add(const Grid* other, double scale, double offset) = 0;
  virtual size_t get_grid_size() const = 0;
  virtual void one2multi(size_t index, size_t* result) const = 0;
  virtual double expected_bias() const = 0;
  //clear all values and derivatives
  virtual void clear() = 0;

};

template< int DIM>
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

  /**
   * Constructor from file, with interpolation specified
   **/
 DimmedGrid(const std::string& input_grid, int b_interpolate): b_derivatives_(0), b_interpolate_(b_interpolate), grid_(NULL), grid_deriv_(NULL) {
    read(input_grid);
  }

  /**
   * Constructor from grid file
   **/
 DimmedGrid(const std::string& input_grid): b_derivatives_(0), b_interpolate_(1), grid_(NULL), grid_deriv_(NULL) {
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
      if(b_periodic_[i]){
	xi -= (max_[i] - min_[i]) * int_floor((xi - min_[i]) / (max_[i] - min_[i]));
      }
      result[i] = (size_t) floor((xi - min_[i]) / dx_[i]);
    }
  }

  void add(const Grid* other, double scale, double offset) {
    //probably should specialize here, but I'll blindly assume the dimensions are the same
    size_t index[DIM];
    double x[DIM];
    double der[DIM];
    size_t i,j;
    for(i = 0; i < grid_size_; i++) {
      one2multi(i, index);
      for(j = 0; j < DIM; j++) {
	x[j] = min_[j] + dx_[j] * index[j];
      }
      grid_[i] += scale * other->get_value_deriv(x, der) + offset;
      for(j = 0; j < DIM; j++)
	grid_deriv_[i*DIM + j] += scale * der[j];
    }
  }
  
  double max_value() const {
    double max = grid_[0];
    size_t i;
    for(i = 0; i < grid_size_; i++) {
      max = fmax(max, grid_[i]);
    }
    
    return max;
  }

  double min_value() const {
    double min = grid_[0];
    size_t i;
    for(i = 0; i < grid_size_; i++) {
      min = fmin(min, grid_[i]);
    }
    return min;
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
    int i;

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

    if(!in_grid(x)) {
//	std::cout << "THIS SHOULD BE PRINTING\n";
      /*
      std::cerr << "Bad grid value (";
      size_t i;
      for(i = 0; i < DIM; i++)
	std::cerr << x[i]  << " (" << min_[i]  << "," << max_[i] << ")" << ", ";
      std::cerr << ")" << std::endl;
      */
      return 0;
    }

      
    if(b_interpolate_ && b_derivatives_) {
      double temp[DIM];
      return get_value_deriv(x, temp);
    }

    size_t index[DIM];
    get_index(x, index);
    return grid_[multi2one(index)];
  }

  /**
   * Add a value to the grid. Only makes sense if there is no derivative
   **/
   double add_value(const double* x0, double value) {
    if(b_interpolate_) {
      edm_error("Cannot add_value when using derivatives", "grid.h:add_value");
    }

    if(!in_grid(x0)) {
      return 0;
    }

    size_t index[DIM];
    size_t index1;
    get_index(x0, index);
    index1 = multi2one(index);
    grid_[index1] += value;    
    return value;
  }

  /**
   * Get value and derivatives, optionally using interpolation
   **/ 
   double get_value_deriv(const double* x, double* der) const {
    
    double value;
    size_t index[DIM];
    size_t index1;
    size_t i;
    
    //checks
    if(!in_grid(x)) {
      /*
      std::cerr << "Bad grid value (";
      size_t i;
      for(i = 0; i < DIM; i++)
	std::cerr << x[i]  << " (" << min_[i]  << "," << max_[i] << ")" << ", ";
      std::cerr << ")" << std::endl;
      */
	//std::cout << "YOU SHOULD READ THIS FOR interpolation_1d\n";
      for(i = 0; i < DIM; i++)
	der[i] = 0;
      return 0;
    }

    get_index(x, index);
    index1 = multi2one(index);

    if(b_interpolate_) {
      
      double where[DIM]; //local position (local meaning relative to neighbors)
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
  }
   
  void write(const std::string& filename) const {
    using namespace std;
    ofstream output;
    size_t i, j;
    output.open(filename.c_str());

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
    size_t temp[DIM];
    for(i = 0; i < grid_size_; i++) {
      one2multi(i, temp);
      for(j = 0; j < DIM; j++) {
	output << setprecision(8) << std::fixed << (min_[j] + dx_[j] * temp[j]) << " ";
      }
      output << setprecision(8) << std::fixed << grid_[i] << " ";
      if(b_derivatives_) {
	for(j = 0; j < DIM; j++) {
	  output << setprecision(8) << std::fixed <<  -grid_deriv_[i*DIM + j] << " ";
	}
      }
      output << endl;
      if(temp[0] == grid_number_[0] - 1)
	output << endl;
    }

    output.close();    
  }

  /**
   * MPI-based writing that will attempt to gather all grid values. Needs to know 
   * overall box size, since MPI processes don't normally know this.
   **/
  void multi_write(const std::string& filename, const double box_min[DIM], 
		   const double box_max[DIM], 
		   const int b_periodic[DIM],
		   int b_lammps_format) const {

    using namespace std;

    if(b_lammps_format == 1 && DIM > 1){
      edm_error("Lammps format only valid for 1D grids", "grid.h:multi_write");
    }
        
    unsigned int i, j;
    int myrank, otherrank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double x[DIM];
    unsigned int reduced_counts[DIM];
    ofstream output;     
    double der[DIM];
    int b_amwriting = 0;
    size_t super_index[DIM];
    size_t temp;
    double value;
    unsigned int extra_n = 0;

    //calculate the extra lines, if any we need to create
    if(b_lammps_format)
      extra_n = box_min[0] / dx_[0];	


    
    //first, count the number of points that we will have in this box
    for(i = 0; i < DIM; i++) {      
      reduced_counts[i] = (int) ceil((box_max[i] - box_min[i]) / dx_[i]);      
      reduced_counts[i] = b_periodic[i] ? reduced_counts[i] : reduced_counts[i] + 1;
    }       
    
    //write header
    if(myrank == 0) {
      
      output.open(filename.c_str());
      
      if(!b_lammps_format) {
	// print plumed-style header
	output << "#! FORCE " << b_derivatives_ << endl;
	output << "#! NVAR " << DIM << endl;
	
	output << "#! TYPE ";
	for(i = 0; i < DIM; i++)
	  output << GRID_TYPE << " ";
	output << endl;
	
	output << "#! BIN ";
	for(i = 0; i < DIM; i++) 
	  output << (b_periodic[i] ? reduced_counts[i] : reduced_counts[i] - 1 ) << " ";
	output << endl;
	
	output << "#! MIN ";
	for(i = 0; i < DIM; i++)
	  output << box_min[i] << " ";
	output << endl;
	
	output << "#! MAX ";
	for(i = 0; i < DIM; i++)
	  output << box_max[i] << " ";
	output << endl;
	
	output << "#! PBC ";
	for(i = 0; i < DIM; i++)
	  output << b_periodic[i] << " ";
	output << endl;
	
      } else {
	output << "#Auto generated by electronic-dance-music" << endl << endl;	
	output << "EDM" << endl;
	output << "N " << extra_n + reduced_counts[0] 
	       << " R " << dx_[0] << " " 
	       << box_max[0] << endl << endl;
	//fill lines until the bias starts
	for(i = 1; i < extra_n; i++)
	  output << i << " " << i * dx_[0] << " 0.0" << " 0.0" << endl;

      }
      
      output.close();

    }
    
    size_t total = 1;
    for(i = 0; i < DIM; i++)
      total *= reduced_counts[i];
    
    //print out grid
    for(i = 0; i < total; i++) {

      //one2multi
      temp = i;
      for(j = 0; DIM > 1 && j < DIM-1; j++) {
	super_index[j] = temp % reduced_counts[j];
	temp = (temp - super_index[j]) / reduced_counts[j];
	x[j] = super_index[j] * dx_[j] +  box_min[j];
      }
      super_index[j] = temp; 
      x[j] = super_index[j] * dx_[j] +  box_min[j];

      if(in_grid(x)) {

	//sort out who is going to write, since there is overlap possible		
	MPI_Allreduce(&myrank, &otherrank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		
	if(!b_amwriting && otherrank == myrank) {
	  //	  cout << "[" << i << "] I am " << myrank << " and I'm going to write now" << endl;
	  b_amwriting = 1;
	  output.open(filename.c_str(), std::fstream::out | std::fstream::app);
	} else if(b_amwriting && otherrank != myrank) {

	  //	  cout << "[" << i << "] I am " << myrank << "and I will stop writing, due to outrank" << endl;
	  b_amwriting = 0;
	  output.close();

	}
      } else {
	
	//to synchronize
	otherrank = -1;
	//use value to throw away result
	MPI_Allreduce(&otherrank, &value, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	if(b_amwriting) {
	  //	  cout << "[" << i << "] I am " <<  myrank << "and I will stop, due to out of grid" << endl;
	  b_amwriting = 0;
	  output.close();
	}
	
      }
      
      MPI_Barrier(MPI_COMM_WORLD);

      //now write!
      if(b_amwriting) {
	if(b_lammps_format)
	  output << i + extra_n << " ";
	for(j = 0; j < DIM; j++) {
	  output << setprecision(8) << std::fixed << x[j] << " ";
	}
	if(b_derivatives_)
	  value = get_value_deriv(x, der);
	else
	  value = get_value(x);
	output << setprecision(8) << std::fixed << value << " ";
	if(b_derivatives_) {
	  for(j = 0; j < DIM; j++) {
	    output << setprecision(8) << std::fixed <<  -der[j] << " ";
	  }
	}

	output << endl;
	if(super_index[0] == reduced_counts[0] - 1)
	  output << endl;
      }
    }
    if(b_amwriting)
      output.close();
  }

  /**
   * Clears all values and derivatives
   **/
  void clear() {
    int i, j;
    for(i = 0; i < grid_size_; i++) {
      grid_[i] = 0;
      if(b_derivatives_)
	for(j = 0; j < DIM; j++)
	  grid_deriv_[i * DIM + j] = 0;
    }
    
  }

  //calculate the expected bias assuming the grid is made 
  //up of unormalized -ln(p)
  double expected_bias() const {
    size_t i;
    double Z, offset, avg;
    Z = offset = avg = 0;

    //make sure the highest value is 0
    for(i = 0; i < grid_size_; i++)
      offset = fmax(offset, grid_[i]);

    //get partition coefficient
    for(i = 0; i < grid_size_; i++)
      Z += exp(-grid_[i] - offset);

    //integrate
    for(i = 0; i < grid_size_; i++)
      avg += grid_[i]  * exp(-grid_[i] - offset);
    
    return avg / Z;
  }
  
  void read(const std::string& filename) {
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
      free(grid_);
      grid_ = NULL;
    }
    if(grid_deriv_ != NULL){
      free(grid_deriv_);
      grid_deriv_ = NULL;
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

  void set_interpolation(int b_interpolate) {
    b_interpolate_ = b_interpolate;
  }
 
  double* get_grid() {
    return grid_;
  }

  const double* get_dx() const{
    return const_cast<double*>(dx_);
  }

  const double* get_min() const{
    return const_cast<double*>(min_);
  }

  const double* get_max() const{
    return const_cast<double*>(max_);
  }


  size_t get_grid_size() const{
    return grid_size_;
  }

  /**
   * Check if a point is in bounds
   **/
   bool in_grid(const double x[DIM]) const {
    size_t i;
    for(i = 0; i < DIM; i++) {
	if(!b_periodic_[i] && (x[i] < min_[i] || x[i] > (max_[i])) ){
	return false;
      }
    }
    return true;
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
  virtual void initialize() {//virtual to allow GPU override
    size_t i;
    grid_size_ = 1;
    for(i = 0; i < DIM; i++)
      grid_size_ *= grid_number_[i];
    grid_ = (double *) calloc( grid_size_, sizeof(double));
    if(b_derivatives_) {
      grid_deriv_ = (double *) calloc( DIM * grid_size_, sizeof(double));
      if(!grid_deriv_) {
	edm_error("Out of memory!!", "grid.h:initialize");	
      }
    }
  }  
};


/**
 * This is a non-template constructor which dispatches to the appropiate template
 **/
Grid* make_grid( int dim, 
		const double* min, 
		const double* max, 
		const double* bin_spacing, 
		const int* b_periodic, 
		int b_derivatives, 
		int b_interpolate);

/**
 * This is a non-template constructor which dispatches to the appropiate template
 **/

Grid* read_grid( int dim, const std::string& filename, int b_interpolate);

/**
 * This is a non-template constructor which dispatches to the appropiate template
 **/
Grid* read_grid( int dim, const std::string& filename);


}
#endif //GRID_H_
