#ifndef GAUSS_GRID_H_
#define GAUSS_GRID_H_

#include "grid.h"
#include <string>
#include <iostream>
#include <cmath>

#define GAUSS_SUPPORT 6.25 // sigma^2 considered for gaussian

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
  virtual double add_gaussian(const double* x, double height) = 0;
  virtual void set_boundary(const double* min, const double* max, const int* b_periodic) = 0;
  virtual double get_volume() const = 0;
  virtual int in_bounds(const double* x) const = 0;
  virtual void multi_write(const std::string& filename) const = 0;
};

template<int DIM>
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
		 const double* sigma) : grid_(min, max, bin_spacing, b_periodic, 1, b_interpolate){
    //the 1 means we always use derivatives for a gaussian grid
    
    size_t i;
    for(i = 0; i < DIM; i++) {
      sigma_[i] = sigma[i];
      boundary_min_[i] = min[i];
      boundary_max_[i] = max[i];
      b_periodic_boundary_[i] = b_periodic[i];
    }
    
    update_minigrid();
  }

 DimmedGaussGrid(const std::string& filename, const double* sigma) : grid_(filename) {
    size_t i;
    for(i = 0; i < DIM; i++) {
      sigma_[i] = sigma[i];
      boundary_min_[i] = grid_.min_[i];
      boundary_max_[i] = grid_.max_[i];
      b_periodic_boundary_[i] = 0;
    }
    
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
    grid_.multi_write(filename, boundary_min_, boundary_max_, b_periodic_boundary_);
  }

  virtual void multi_write(const std::string& filename, 
			   const double* box_low, 
			   const double* box_high, 
			   const int* b_periodic) const {
    grid_.multi_write(filename, box_low, box_high, b_periodic);
  }

  
  void set_interpolation(int b_interpolate) {
    grid_.b_interpolate_ = b_interpolate;
  }

  double add_gaussian(const double* x0, double height) {

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
    double vol_element = 1;

    //get volume element for bias integration
    for(i = 0; i < DIM; i++) {
      vol_element *= grid_.dx_[i]; 
    }

    //switch to non-const so we can wrap
    double x[DIM];
    for(i = 0; i < DIM; i++)
      x[i] = x0[i];


    remap(x); //attempt to remap to be close or in grid

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
      index1 = i - minisize_total_ / 2;
      for(j = 0; j < DIM-1; j++) {
	index[j] = index1 % minisize_[j];
	index1 = (index1 - index[j]) / static_cast<long int>(minisize_[j]);
      }
      index[j] = index1; 


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

      dp2 *= 0.5;
      
      if(dp2 < GAUSS_SUPPORT) {
	expo = height * exp(-dp2);
      
	//actually add hill now!
	xx_index1 = grid_.multi2one(xx_index);
	grid_.grid_[xx_index1] += expo;      
	bias_added += expo * vol_element;
	//and POSITIVE derivative!! (different than plumed)
	for(j = 0; j < DIM; j++) {
	  grid_.grid_deriv_[(xx_index1) * DIM + j] -= dp[j] / sigma_[j] * expo;
	}
      }
    }
    return bias_added;
  }

  /**
   * Specifying the period here means that we can wrap points along
   * the boundary, not necessarily along the grid bounds
   **/
  void set_boundary(const double* min, const double* max, const int* b_periodic) {
    size_t i;
    for(i = 0; i < DIM; i++) {
      boundary_min_[i] = min[i];
      boundary_max_[i] = max[i];
      b_periodic_boundary_[i] = b_periodic[i];
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
	if(fabsl(grid_.min_[i] - x[i] - dp[0]) < fabsl(grid_.max_[i] - x[i] - dp[1]))
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

 private:
  /**
   * Calculate the amount of grid that needs to be considered based on gaussian width and grid width 
   **/
  void update_minigrid() {
    size_t i;
    double dist;

    minisize_total_ = 1;
    for(i = 0; i < DIM; i++) {
      dist = sqrt(2 * GAUSS_SUPPORT) * sigma_[i]; //a distance that goes for gaussian center outwards
      minisize_[i] = int_floor(dist / grid_.dx_[i]) + 1;
      minisize_total_ *= (2 * minisize_[i] + 1); // the minisize is only in 1 direction, but we in total must look forward and back from center
    }
  }
  
};


GaussGrid* make_gauss_grid(unsigned int dim, 
			   const double* min, 
			   const double* max, 
			   const double* bin_spacing, 
			   const int* b_periodic, 
			   int b_interpolate,
			   const double* sigma);


GaussGrid* read_gauss_grid(unsigned int dim, const std::string& filename, const double* sigma);


#endif //GAUSS_GRID_H_
