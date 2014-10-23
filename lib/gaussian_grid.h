#ifndef GAUSS_GRID_H_
#define GAUSS_GRID_H_

#include "grid.h"
#include <string>
#include <iostream>
#include <cmath>

#define GAUSS_SUPPORT 6. // sigma^2 considered for gaussian

/**
 * This class uses a compositional ("has-a") relationship with Grid. This is the 
 * interface and DimmedGaussGrid has the dimension template parameter
 **/
class GaussGrid : public Grid {
  virtual void add_gaussian(const double* x, double height) = 0;
  virtual void set_boundary(const double* min, const double* max) = 0;
  virtual double get_volume() const = 0;
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
		 const double* sigma) : grid_(min, max, bin_spacing, b_periodic, 1, b_interpolate) {
    //the 1 means we always use derivatives for a gaussian grid
    
    size_t i;
    for(i = 0; i < DIM; i++) {
      sigma_[i] = sigma[i];
      boundary_min_[i] = min[i];
      boundary_max_[i] = max[i];
    }
    
    update_minigrid();
  }

  double get_value(const double* x) const {
    if(!in_bounds(x))
      return 0;

    return grid_.get_value(x);
  }

  double get_value_deriv(const double* x, double* der) const{

    size_t i;
    if(!in_bounds(x)) {
      for(i = 0; i < DIM; i++)
	der[i] = 0;
      return 0;
    }
    return grid_.get_value_deriv(x, der);
  }

  void read(const std::string& filename) {
    grid_.read(filename);
  }

  void write(const std::string& filename) const {
    grid_.write(filename);
  }
  
  void set_interpolation(int b_interpolate) {
    grid_.b_interpolate_ = b_interpolate;
  }

  void add_gaussian(const double* x, double height) {
    //check if we need to add
    if(!in_bounds(x))
      return;

    size_t i,j;

    int index[DIM];//some temp local index, possibly negative
    int index1; //some temp collapsed index, possibly negative

    double xx[DIM]; //points away from hill center but affected by addition
    size_t xx_index[DIM];//The grid index that corresponds to xx
    size_t xx_index1;//The collapsed grid index that corresponds to xx
    size_t x_index[DIM];//The grid index that corresponds to the hill center
    size_t x_index1;//The collapsed grid index that corresponds to the hill center
    int b_flag; //a flag
    double dp[DIM]; //essentially distance vector, changes in course of calculation
    double dp2; //essentially magnitude of distance vector, changes in course of calculation
    double expo; //exponential portion used in calculation

    grid_.get_index(x, x_index);
    x_index1 = grid_.multi2one(x_index);

    //loop over only the support of the gaussian
    for(i = 0; i < minisize_total_; i++) {
      
      //get offset of current point by converting the i to an index
      //This is the one2multi algorithm

      //We substract here so that we consider both below and above hill center
      index1 = i - minisize_total_ / 2;
      for(j = 0; j < DIM-1; j++) {
	index[j] = index1 % minisize_[j];
	index1 = (index1 - index[j]) / minisize_[j];
      }
      index[j] = index1; 


      b_flag = 0;
      //convert offset into grid index and point
      for(j = 0; j < DIM; j++) {
	//turn offset into index
	xx_index[j] = (index[j] + x_index[j]);
	//check if out of grid or wrap
	if(xx_index[j] >= grid_.grid_number_[j]) {
	  if(grid_.b_periodic_[j]) {
	    xx_index[j] %= grid_.grid_number_[j];
	  } else {
	    b_flag = 1; //we don't need to consider this point
	    break;
	  }
	}

	xx[j] = grid_.min_[j] + grid_.dx_[j] * xx_index[j];
	
	//is this point within the boundary?
	if(xx[j] < boundary_min_[j] || xx[j] > boundary_max_[j]) {
	  b_flag = 1;
	  break;
	}
      }

      //was this point out of grid or boundary?
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
      }
      
      //actually add hill now!
      xx_index1 = grid_.multi2one(xx_index);
      grid_.grid_[xx_index1] += expo;
      //and derivative
      for(j = 0; j < DIM; j++) {
	grid_.grid_deriv_[(xx_index1) * DIM + j] = dp[j] / sigma_[j] * expo;
      }
    }    
  }

  void set_boundary(const double* min, const double* max) {
    size_t i;
    for(i = 0; i < DIM; i++) {
      boundary_min_[i] = min[i];
      boundary_max_[i] = max[i];
    }
    
  }

  double get_volume() const {
    double vol = 1;
    size_t i;
    for(i = 0; i < DIM; i++) {
      vol *= boundary_max_[i] - boundary_min_[i];
    }
  }
 
  double* get_grid() {
    return grid_.grid_;
  }

  size_t get_grid_size() const{
    return grid_.grid_size_;
  }



  size_t minisize_[DIM];// On DIM-dimensional grid, how far we must search before gaussian decays enough to ignore
  size_t minisize_total_; //On reduced grid, how far we must search before gaussian decays enough to ignore
  double sigma_[DIM];//gaussian sigma
  double boundary_min_[DIM]; //optional boundary minmimum
  double boundary_max_[DIM]; //optional boundary maximum
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

  int in_bounds(const double x[DIM]) const {

    size_t i;
    for(i = 0; i < DIM; i++) {
      if(x[i] < boundary_min_[i] || x[i] > boundary_max_[i])
	return 0;
    }

    return 1;
  }

  
};

#endif //GAUSS_GRID_H_
