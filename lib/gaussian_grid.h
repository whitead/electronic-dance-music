#ifndef GAUSS_GRID_H_
#define GAUSS_GRID_H_

#include "grid.h"
#include "edm.h"
#include <string>
#include <iostream>
#include <cmath>

#define GAUSS_SUPPORT 8.0 // sigma^2 considered for gaussian
#define BC_TABLE_SIZE 65536 //boundary correction function look up size 
#define BC_MAR 2.0
#define BC_CORRECTION


inline
double sigmoid(double x) {
  if(x < 0)
    return 1;
  if(x > 1)
    return 0;
  return 2*x*x*x - 3*x*x + 1;
}

inline
double sigmoid_dx(double x) {
  if(x < 0)
    return 0;
  if(x > 1)
    return 0;
  return 6*x*x - 6*x;
}


namespace EDM{

/**
 * This class uses a compositional ("has-a") relationship with Grid. This is the 
 * interface and DimmedGaussGrid has the dimension template parameter
 **/
  class GaussGrid : public Grid {
    /**
     * Retruns the integrated amount of bias added to the system
     **/
  public:
    virtual ~GaussGrid() {};
    virtual  double add_value(const double* x, double height) = 0;
    virtual void set_boundary(const double* min, const double* max, const int* b_periodic) = 0;
    virtual double get_volume() const = 0;
    virtual  int in_bounds(const double* x) const = 0;
    virtual void multi_write (const std::string& filename) const = 0;
    virtual void multi_write(const std::string& filename, 
			     const double* box_low, 
			     const double* box_high, 
			     const int* b_periodic,
			     int b_lammps_format) const = 0;
    /**
     *Write out the file in lammps tabular potential format.
     **/
    virtual void lammps_multi_write(const std::string& filename) const = 0;
  };

  template< int DIM>
    class DimmedGaussGrid : public GaussGrid{
    /** A class for treating grids that have gaussians on it 
     *
     *
     **/
  public:
  DimmedGaussGrid(const double* min, 
		  const double* max, 
		  const double* bin_spacing, 
		  const int* b_periodic, 
		  int b_interpolate,
		  const double* sigma) : grid_(min, max, bin_spacing, b_periodic,
					       1, b_interpolate), b_dirty_bounds(0){
      //the 1 means we always use derivatives for a gaussian grid
    
      size_t i;
      for(i = 0; i < DIM; i++) {
	sigma_[i] = sigma[i] * sqrt(2.);
      }
    
      set_boundary(min, max, b_periodic);
      update_minigrid();
    }

    /*
     *Default constructor
     */
//  DimmedGaussGrid(): b_interpolate_(1), grid_(NULL), grid_deriv_(NULL){}


    /**
     * Rebuild from a file. Files don't store sigma, so it must be set again.
     **/
  DimmedGaussGrid(const std::string& filename, const double* sigma) : grid_(filename), b_dirty_bounds(0) {
      size_t i;
      for(i = 0; i < DIM; i++) {
	sigma_[i] = sigma[i] * sqrt(2.);
      }

      set_boundary(grid_.min_, grid_.max_, grid_.b_periodic_);    
      update_minigrid();
    }

    ~DimmedGaussGrid() {
      //nothing
    }

    double get_value(const double* x) const {

      size_t i;

      //for constness
      double xx[DIM];
      for(i = 0; i < DIM; i++)
	xx[i] = x[i];

      //Attempt to wrap around the specified boundaries (possibly separate from grid bounds)
      if(!in_bounds(xx)) {
	remap(xx);
	if(!in_bounds(xx))
	  return 0;
      }
    
      return grid_.get_value(xx);
    }

    double get_value_deriv(const double* x, double* der) const{

      size_t i;

      //for constness
      double xx[DIM];
      for(i = 0; i < DIM; i++)
	xx[i] = x[i];

      //Attempt to wrap around the specified boundaries (separate from grid bounds)
      if(!in_bounds(xx)) {
	remap(xx);
	if(!in_bounds(xx)) {
	  for(i = 0; i < DIM; i++)
	    der[i] = 0;
	  return 0;
	}
      }

      return grid_.get_value_deriv(xx, der);
    }

    void read(const std::string& filename) {
      grid_.read(filename);
    }

    void write(const std::string& filename) const {
      grid_.write(filename);
    }

    /**
     * Uses its known boundaries to handle assembling all grids
     **/
    void multi_write(const std::string& filename) const {
      grid_.multi_write(filename, boundary_min_, boundary_max_, b_periodic_boundary_, 0);
    }

    void lammps_multi_write(const std::string& filename) const {
      grid_.multi_write(filename, boundary_min_, boundary_max_, b_periodic_boundary_, 1);
    }


    void multi_write(const std::string& filename, 
		     const double* box_low, 
		     const double* box_high, 
		     const int* b_periodic,
		     int b_lammps_format) const {
      grid_.multi_write(filename, box_low, box_high, b_periodic, 0);
    }

  
    void set_interpolation(int b_interpolate) {
      grid_.b_interpolate_ = b_interpolate;
    }

    /**
     * The workhorse method of the program. The source is very well-documented
     **/
    double  add_value(const double* x0, double height) {

      size_t i,j;

      int index[DIM];//some temp local index, possibly negative
      int index1; //some temp collapsed index, possibly negative

      double xx[DIM]; //points away from hill center but affected by addition
      size_t xx_index[DIM];//The grid index that corresponds to xx
      size_t xx_index1;//The collapsed grid index that corresponds to xx
      int x_index[DIM];//The grid index that corresponds to the hill center, possibly negative for points outside grid
      int b_flag; //a flag
      double dp[DIM]; //essentially distance vector, changes in course of calculation
      double dp2; //essentially magnitude of distance vector, changes in course of calculation
      double expo; //exponential portion used in calculation
      double bias_added = 0;// amount of bias added to the system as a result. decreases due to boundaries
      double vol_element = 1;//integration volume element

      double bc_denom; //Boundary correction denominator
      double bc_correction;
      double bc_force[DIM]; //Boundary correction force denominator
      size_t bc_index; //Boundary correction index
      double temp1, temp2, temp3, temp4, temp5, temp6, temp7;

      //get volume element for bias integration
      for(i = 0; i < DIM; i++) {
	vol_element *= grid_.dx_[i]; 
      }

      //switch to non-const so we can wrap
      double x[DIM];
      for(i = 0; i < DIM; i++)
	x[i] = x0[i];


      remap(x); //attempt to remap to be close or in grid

      //now do check on if we are in the boundary of the overall grid
      for(i = 0; i < DIM; i++)
	if(!b_periodic_boundary_[i] && (x[i] < boundary_min_[i] || x[i] > boundary_max_[i]))
	  return 0;


      //now we are at the closest possible image, find an index            
      //normally, would be grid_.get_index(x, x_index);
      //but we need to consider negative indices    
      for(i = 0; i < DIM; i++) {
	x_index[i] = int_floor((x[i] - grid_.min_[i]) / grid_.dx_[i]);
      }

      //loop over only the support of the gaussian
      for(i = 0; i < minisize_total_; i++) {
      
	//get offset of current point by converting the i to an index
	//This is the one2multi algorithm

	//We substract here so that we consider both below and above hill center
	index1 = i;
	for(j = 0; DIM > 1 && j < DIM-1; j++) {
	  index[j] = index1 % (2 * minisize_[j] + 1);
	  index1 = (index1 - index[j]) / static_cast<long int>(2 * minisize_[j] + 1);
	}
	index[j] = index1;

	for(j = 0; j < DIM; j++)
	  index[j] -= minisize_[j];


	b_flag = 0;
	//convert offset into grid index and point
	for(j = 0; j < DIM; j++) {
	  //turn offset into index
	  index[j] += x_index[j];

	  //check if out of grid or needs to be wrap
	  if(index[j] >= grid_.grid_number_[j]) {
	    if(grid_.b_periodic_[j]) {
	      index[j] %= grid_.grid_number_[j];
	    } else {
	      b_flag = 1; //we don't need to consider this point
	      break;
	    }
	  }
	  if(index[j] < 0) {
	    if(grid_.b_periodic_[j]) {
	      index[j] += grid_.grid_number_[j];
	    } else {
	      b_flag = 1; //we don't need to consider this point
	      break;
	    }
	  }
	  //we know now it's > 0 and in grid
	  xx_index[j] = static_cast<size_t>(index[j]);

	  xx[j] = grid_.min_[j] + grid_.dx_[j] * xx_index[j];
	
	  //is this point within the boundary?
	  if(!b_periodic_boundary_[j] && (xx[j] < boundary_min_[j] || xx[j] > boundary_max_[j])) {
	    b_flag = 1;
	    break;
	  }
	}

	//was this point out of grid?
	if(b_flag)
	  continue;

      
	//nope, it's a valid point we need to put gaussian mass on 
	dp2 = 0;
	for(j = 0; j < DIM; j++) {
	  dp[j] = xx[j] - x[j];
	  //wrap this distance
	  if(grid_.b_periodic_[j]) {
	    dp[j] -= round(dp[j] / (grid_.max_[j] - grid_.min_[j])) * (grid_.max_[j] - grid_.min_[j]);
	  }
	  //scale by sigma
	  dp[j] /= sigma_[j];
	  dp2 += dp[j] * dp[j];
	}

	//This no longer happens. It is accounted for during file read and construction
	//      dp2 *= 0.5;
	if(dp2 < GAUSS_SUPPORT) {
	  expo = exp(-dp2);

	  //treat boundary corrected hill if necessary
	  bc_denom = 1.0;	
	  bc_correction = 0;
	  for(j = 0; j < DIM; j++) {
	    if(!b_periodic_boundary_[j]) {
	      //this will automatically cast to int
	      bc_index = (BC_TABLE_SIZE - 1) * (xx[j] - boundary_min_[j]) / (boundary_max_[j] - boundary_min_[j]);

	      temp1 = exp(-pow(x[j] - boundary_min_[j], 2) / (pow(sigma_[j],2)));
	      temp2 = sigmoid((xx[j] - boundary_min_[j]) / (sigma_[j] * BC_MAR));
	      temp3 = exp(-pow(x[j] - boundary_max_[j], 2) / (pow(sigma_[j],2)));
	      temp4 = sigmoid((boundary_max_[j] - xx[j]) / (sigma_[j] * BC_MAR));

#ifdef BC_CORRECTION
	      bc_correction = (temp1  - expo ) * temp2 + (temp3 - expo ) * temp4;
#endif
	      bc_denom *= bc_denom_table_[j][bc_index];
	    
	      //dp has been divided by sigma once already
	      temp5 = -2 * dp[j] / sigma_[j];
	      temp6 = sigmoid_dx((xx[j] - boundary_min_[j]) / (sigma_[j] * BC_MAR)) / (BC_MAR * sigma_[j]);
	      temp7 = -sigmoid_dx((boundary_max_[j] - xx[j]) / (sigma_[j] * BC_MAR)) / (BC_MAR * sigma_[j]);
	    
	      //this is just the force of the uncorrected
	      bc_force[j] = temp5 * expo;

#ifdef BC_CORRECTION
	      bc_force[j] +=  (temp1 - expo) * temp6 - 
		temp5 * expo * temp2 + 
		(temp3 - expo) * temp7  -
		temp5 * expo * temp4;
#endif

	      bc_force[j] = bc_force[j] * bc_denom - bc_denom_deriv_table_[j][bc_index] * (expo + bc_correction);	    
	      bc_force[j] /= bc_denom * bc_denom;
	      bc_correction /= bc_denom;
	    
	    } else {
	      bc_denom *=  sqrt(M_PI) * sigma_[j];
	    }
	  }
	  expo /= bc_denom;
	
      
	  //actually add hill now!
	  xx_index1 = grid_.multi2one(xx_index);
	  grid_.grid_[xx_index1] += height * (expo + bc_correction);
	  bias_added += height * (expo + bc_correction) * vol_element;
	  for(j = 0; j < DIM; j++) {
	    if(b_periodic_boundary_[j])
	      grid_.grid_deriv_[(xx_index1) * DIM + j] -= height * (2 * dp[j] / sigma_[j] * expo);
	    else
	      grid_.grid_deriv_[(xx_index1) * DIM + j] += height * bc_force[j];
	  }

	  if(!b_dirty_bounds && bc_correction * bc_correction >  0)
	    b_dirty_bounds = 1; //set it to be true that our bounds are dirty.

	}
      }

      //If we have added gaussians and are trying to maintain 
      //out of bounds with 0 derivative, we need to update the out of bounds potential
      if(b_dirty_bounds) {
	duplicate_boundary();
	b_dirty_bounds = 0;
      }


      return bias_added;
    }

    /**
     * Specifying the period here means that we can wrap points along
     * the boundary, not necessarily along the grid bounds
     **/
    void set_boundary(const double* min, const double* max, const int* b_periodic) {
      size_t i,j;
      double s;
      double tmp1,tmp2,tmp3;

      b_dirty_bounds = 0;

      for(i = 0; i < DIM; i++) {
	boundary_min_[i] = min[i];
	boundary_max_[i] = max[i];
	b_periodic_boundary_[i] = b_periodic[i];
      }

      //pre-compute bc boundaries if necessary
      for(i = 0; i < DIM; i++) {
	if(!b_periodic_boundary_[i]) {
	  for(j = 0; j < BC_TABLE_SIZE; j++) {
	    s = j * (boundary_max_[i] - boundary_min_[i]) / (BC_TABLE_SIZE - 1) + boundary_min_[i];

	    //mcgovern-de pablo contribution
	    tmp1 = sqrt(M_PI) * sigma_[i] / 2.  * 
	      ( erf((s - boundary_min_[i]) / sigma_[i]) + 
		erf((boundary_max_[i] - s) / sigma_[i]));

	    bc_denom_table_[i][j] = tmp1;
#ifdef BC_CORRECTION
	    tmp2 = sqrt(M_PI) * sigma_[i] / 2. * erf((boundary_max_[i] - boundary_min_[i]) / sigma_[i]);

	    bc_denom_table_[i][j] += (tmp2 - tmp1) * 
	      sigmoid((s - boundary_min_[i]) / (BC_MAR * sigma_[i]));
	    bc_denom_table_[i][j] += (tmp2 - tmp1) * 
	      sigmoid((boundary_max_[i] - s) / (BC_MAR * sigma_[i]));
#endif
	    //mcgovern-de pablo contribution derivative
	    tmp3 = 1. * 
	      (exp( -pow(s - boundary_min_[i],2) / pow(sigma_[i],2)) - 
	       exp( -pow(boundary_max_[i] - s,2)/ pow(sigma_[i],2)));

	    bc_denom_deriv_table_[i][j] = tmp3;
#ifdef BC_CORRECTION
	    bc_denom_deriv_table_[i][j] += (tmp2 - tmp1) * 
	      sigmoid_dx((s - boundary_min_[i]) / (BC_MAR * sigma_[i])) / (BC_MAR * sigma_[i]) - 
	      tmp3 * sigmoid((s - boundary_min_[i]) / (BC_MAR * sigma_[i]));
	    bc_denom_deriv_table_[i][j] += -(tmp2 - tmp1) * 
	      sigmoid_dx((boundary_max_[i] - s) / (BC_MAR * sigma_[i])) / (BC_MAR * sigma_[i]) - 
	      tmp3 * sigmoid((boundary_max_[i] - s) / (BC_MAR * sigma_[i]));	  
#endif
	    if(j > 2) {
	      //	    std::cout << ((bc_denom_table_[i][j] - bc_denom_table_[i][j  - 2]) / (2 * (boundary_max_[i] - boundary_min_[i]) / (BC_TABLE_SIZE - 1))) << " =?= " << bc_denom_deriv_table_[i][j-1] << std::endl;
	    }
	    //	  	  bc_denom_table_[i][j] = 1;
	    //	  bc_denom_deriv_table_[i][j] = 0;

	  }
	}
      }
    
    }

    double get_volume() const {
      double vol = 1;
      size_t i;
      for(i = 0; i < DIM; i++) {
	vol *= boundary_max_[i] - boundary_min_[i];
      }
      return vol;
    }
 
    void one2multi(size_t index, size_t result[DIM]) const {
      grid_.one2multi(index, result);
    }

    double* get_grid() {
      return grid_.get_grid();
    }

    const double* get_dx() const{
      return grid_.get_dx();
    }

    const double* get_min() const{
      return grid_.get_min();
    }

    const double* get_max() const{
      return grid_.get_max();
    }

    double max_value() const {
      return grid_.max_value();
    }

    double min_value() const {
      return grid_.min_value();
    }

    void add(const Grid* other, double scale, double offset) {
      grid_.add(other, scale, offset);
    }

    double expected_bias() const {
      return grid_.expected_bias();
    }

    void clear() {
      grid_.clear();
    }
    
    size_t get_grid_size() const{
      return grid_.get_grid_size();
    }

    int in_bounds(const double x[DIM]) const {

      size_t i;
      for(i = 0; i < DIM; i++) {
	if(x[i] < boundary_min_[i] || x[i] > boundary_max_[i])
	  return 0;
      }

      return 1;
    }

    /**
     * Possibly wrap a value across the system boundaries to be as close as possible to the grid
     **/
    void remap(double x[DIM]) const {
    
      double dp[2];
      size_t i;

      //this is a special wrapping. We want to find the nearest image, not the minimal image
      // eg, our grid runs from 0 to 5 and the box runs from 0 to 10.
      // If we want to get the contribution from a particle at 9.5, we need to wrap it to -0.5
      for(i = 0; i < DIM; i++){

	if(x[i] < grid_.min_[i] || x[i] > grid_.max_[i]) {

	  if(grid_.b_periodic_[i]) { //are we periodic, if so than wrap

	    x[i] -= (grid_.max_[i] - grid_.min_[i]) * 
	      int_floor((x[i] - grid_.min_[i]) / 
			(grid_.max_[i] - grid_.min_[i]));

	  } else if(b_periodic_boundary_[i]) { 
	    //if we aren't periodic and out of grid, try wrapping using the system boundaries

	    //find which boundary we can best wrap to, temporarily making use of dp[]	
	    //this is the image choice
	    dp[0] = round((grid_.min_[i] - x[i]) / (boundary_max_[i] - boundary_min_[i])) * 
	      (boundary_max_[i] - boundary_min_[i]);	
	    dp[1] = round((grid_.max_[i] - x[i]) / (boundary_max_[i] - boundary_min_[i])) * 
	      (boundary_max_[i] - boundary_min_[i]);	
	
	    //which bound is closest
	    if(fabs(grid_.min_[i] - x[i] - dp[0]) < fabs(grid_.max_[i] - x[i] - dp[1]))
	      x[i] += dp[0]; //wrap to it
	    else
	      x[i] += dp[1];
	  }
	}
      }
    }


    size_t minisize_[DIM];// On DIM-dimensional grid, how far we must search before gaussian decays enough to ignore
    size_t minisize_total_; //On reduced grid, how far we must search before gaussian decays enough to ignore
    double sigma_[DIM];//gaussian sigma
    double boundary_min_[DIM]; //optional boundary minmimum
    double boundary_max_[DIM]; //optional boundary maximum
    int b_periodic_boundary_[DIM];//optional, this means the explicitly set boundary is treated as periodic
    DimmedGrid<DIM> grid_; //the underlying grid
    double bc_denom_table_[DIM][BC_TABLE_SIZE];
    double bc_denom_deriv_table_[DIM][BC_TABLE_SIZE];


  //these need to be protected b/c we need access to them with the derived GPU class
  protected:
    int b_dirty_bounds; //true if we've added hills and our bounds may be inconsitent. Only needed for simulations where we have 0 derivative forces
    /**
     * Calculate the amount of grid that needs to be considered based on gaussian width and grid width 
     **/
    void update_minigrid() {
      size_t i;
      double dist;

      minisize_total_ = 1;
      for(i = 0; i < DIM; i++) {
	dist = sqrt(2 * GAUSS_SUPPORT) * sigma_[i]; //a distance that goes for gaussian center outwards
	minisize_[i] =  int_floor(dist / grid_.dx_[i]);
	minisize_total_ *= (2 * minisize_[i] + 1);
      }
    }
    
  private:



    void duplicate_boundary() {
    
      size_t i,j,k,l;
      size_t index_outter[DIM], index_bound[DIM];
      size_t min_i[DIM], max_i[DIM];
      int b_flag;

      grid_.get_index(boundary_min_, min_i);
      grid_.get_index(boundary_max_, max_i);

      //move inwards one if necessary
      for(i = 0; i < DIM; i++) {
	while(min_i[i] * grid_.dx_[i] + grid_.min_[i] < boundary_min_[i])
	  min_i[i] += 1; 
	while(max_i[i] * grid_.dx_[i] + grid_.min_[i] > boundary_max_[i] || 
	      max_i[i] == grid_.grid_number_[i])
	  max_i[i] -= 1; 
      }

      //we need to consider the combination of min-1, min, max, max+1 for all points
      size_t offset_size = pow(4,DIM);
      int offset[DIM];
      size_t temp;
      for(i = 0; i < offset_size; i++) {
	b_flag = 0;
	temp = i;
	for(j = 0; j < DIM; j++) {
	  offset[j] = temp % 4;
	  temp = (temp - offset[j]) / 4;
	  switch(offset[j]) {
	  case 0:
	    b_flag |= b_periodic_boundary_[j];
	    b_flag |= min_i[j] == 0;
	    index_outter[j] = min_i[j] - 1;
	    index_bound[j] = min_i[j];
	    break;
	  case 1:
	    index_outter[j] = min_i[j];
	    index_bound[j] = min_i[j];
	    break;
	  case 2:
	    index_outter[j] = max_i[j];
	    index_bound[j] = max_i[j];
	    break;
	  case 3:
	    b_flag |= b_periodic_boundary_[j];
	    b_flag |= max_i[j] == grid_.grid_number_[j] - 1;
	    index_outter[j] = max_i[j] + 1;
	    index_bound[j] = max_i[j];
	    break;
	  }
	}
	if(!b_flag) {
	  //true if this dimension is not periodic or at a boundary
	  k = grid_.multi2one(index_outter);
	  l = grid_.multi2one(index_bound);
	  grid_.grid_[k] = grid_.grid_[l];
	}
      }
    }
  };

/**
 * Used to avoid template constructors
 **/
  GaussGrid* make_gauss_grid( int dim, 
			      const double* min, 
			      const double* max, 
			      const double* bin_spacing, 
			      const int* b_periodic, 
			      int b_interpolate,
			      const double* sigma);

/**
   p * Used to avoid template constructors
**/
  GaussGrid* read_gauss_grid( int dim, const std::string& filename, const double* sigma);

}
#endif //GAUSS_GRID_H_
