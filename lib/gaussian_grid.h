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
  /**
   * Retruns the integrated amount of bias added to the system
   **/
 public:
  virtual double add_gaussian(const double* x, double height) = 0;
  virtual void set_boundary(const double* min, const double* max, const int* b_periodic) = 0;
  virtual double get_volume() const = 0;
  virtual int in_bounds(const double* x) const = 0;
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
      b_periodic_boundary_[i] = 0;
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

  double get_value(const double* x) const {

    size_t i;

    //for constness
    double xx[DIM];
    for(i = 0; i < DIM; i++)
      xx[i] = x[i];

    //Attempt to wrap around the specified boundaries (separate from grid bounds)
    if(!in_bounds(x)) {
      int changed = 0;
      for(i = 0; i < DIM; i++){
	if(b_periodic_boundary_[i]) {
	  xx[i] -= int_floor(xx[i] / (boundary_max_[i] - boundary_min_[i])) * (boundary_max_[i] - boundary_min_[i]);	
	  changed = 1;
	}
      }
      if(!changed || !in_bounds(xx))
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
      int changed = 0;
      for(i = 0; i < DIM; i++){
	if(b_periodic_boundary_[i]) {
	  xx[i] -= int_floor(xx[i] / (boundary_max_[i] - boundary_min_[i])) * (boundary_max_[i] - boundary_min_[i]);	
	  changed = 1;
	}
      }
      if(!changed || !in_bounds(xx)) {
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

  void multi_write(const std::string& filename, const double* box_low, const double* box_high) const {
    grid_.multi_write(filename, box_low, box_high);
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
    size_t x_index[DIM];//The grid index that corresponds to the hill center
    size_t x_index1;//The collapsed grid index that corresponds to the hill center
    int b_flag; //a flag
    double dp[DIM]; //essentially distance vector, changes in course of calculation
    double dp2; //essentially magnitude of distance vector, changes in course of calculation
    double expo; //exponential portion used in calculation
    double bias_added = 0;// amount of bias added to the system as a result. decreases due to boundaries
    double vol_element = 1;

    //switch to non-const so we can wrap
    double x[DIM];
    for(i = 0; i < DIM; i++)
      x[i] = x0[i];
    
    //Attempt to wrap around the specified boundaries (separate from grid bounds)
    if(!in_bounds(x)) {
      for(i = 0; i < DIM; i++){
	if(b_periodic_boundary_[i]) {
	  x[i] -= int_floor(x[i] / (boundary_max_[i] - boundary_min_[i])) * (boundary_max_[i] - boundary_min_[i]);	
	}
      }
    }

    if(!grid_.check_point(x))
      return 0;


    //get volume element for bias integration
    for(i = 0; i < DIM; i++) {
      vol_element *= grid_.dx_[i]; 
    }

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
      bias_added += expo * vol_element;
      //and derivative
      for(j = 0; j < DIM; j++) {
	grid_.grid_deriv_[(xx_index1) * DIM + j] = dp[j] / sigma_[j] * expo;
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


  /* This is a total clusterfuck. some other day
  communicate_soft_boundary(unsigned int* cpu_dim) {

    int neighbors = 1; //number of neighbors, depends only on dimension
    int direction[DIM];//direction we're communicating with 
    int my_cpu_index[DIM];
    int temp;
    size_t i,j,k,l;

    int myrank;
    MPI_Request send_request, rec_request;
    int outgoing, incoming;

    //get my CPU location
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::cout << "I am rank (" << myrank;
    for(i = 0; i < DIM-1; i++) {
      my_cpu_index[i] = myrank % cpu_dim[i];
      myrank = (myrank - my_cpu_index[i]) / cpu_dim[i];
      std::cout << my_cpu_index[i] << ",";
    }
    my_cpu_rank[i] = myrank;

    std::cout << ")" << std:endl;

    //get size for buffer
    size_t buffer_size = 1;
    for(i = 0; i < DIM; i++)
      buffer_size *= soft_boundary_points_[i];

    size_t buffer_size *= (DIM + 1);//each point we need derivative and  value
    double*  send_buffer = (double*) malloc(sizeof(double) * buffer_size);
    double*  rec_buffer = (double*) malloc(sizeof(double) * buffer_size);

    for(i = 0; i < DIM; i++)
      neighbors *= 2;


    //for each neighboring grid
    for(i = 0; i < neighbors; i++) {

      //crate grid index shifts
      temp = i;
      for(j = 0; j < DIM; j++) {
	if(temp % 2 == 0)
	  direction = 1;
	else 
	  direction[i] = -1;

	temp /= 2;
      }

      //now create buffer containing the grid points

      for(j = 0; j < DIM; j++) {
	//for each boundary dimension
	for(k = 0; k < soft_boundary_points_[j]; k++) {
	  //for the points in that dimension

	  //if we're heading left, we go from the boundary to the right
	  //if we're heading right, we go from the boundary to the left
	  if(direction[j] == -1)
	    temp = k;
	  else
	    temp = grid_.grid_number_[i] - k;

	  //the value of the function
	  send_buffer[k * (DIM+1)] = grid_.grid_[temp];
	  grid_.grid_[temp] = 0; //zero it out since we're about to send it
	  //the derivative
	  for(l = 0; l < DIM; l++) {
	    send_buffer[k * (DIM + 1)] = grid._grid_deriv_[temp * DIM + l];
	    grid._grid_deriv_[temp * DIM + l] = 0; //zero it out since we're about to send it
	  }
	}
      }

      //get the rank of the node we're communicating with
      outgoing = my_cpu_rank[DIM - 1] + direction[DIM-1];
      for(j = DIM-1; j > 0; i--)
	outgoing = outgoing * cpu_dim[i- 1] + (my_cpu_rank[i - 1] + direction[i -  1]);

      //send the buffer
      int MPI_Isend(send_buffer, buffer_size, MPI_DOUBLE, outgoing, 0,
		    MPI_COMM_WORLD, &send_request);


      //flip direction to find incoming
      for(j = 0; j < DIM; j++)
	direction[j] *= -1;

      //find incoming
      incoming = my_cpu_rank[DIM - 1] + direction[DIM-1];
      for(j = DIM-1; j > 0; i--)
	incoming = incoming * cpu_dim[i- 1] + (my_cpu_rank[i - 1] + direction[i -  1]);

      
      int MPI_Recv(rec_buffer, buffer_size, MPI_DOUBLE, incoming, 0,
		    MPI_COMM_WORLD, &rec_request);
      
      //unpack the buffer, notice direction has been already flipped
      for(j = 0; j < DIM; j++) {
	//for each boundary dimension
	for(k = 0; k < soft_boundary_points_[j]; k++) {
	  //for the points in that dimension

	  //if we've recieved from our left, we go from the soft boundary to the right into normal region
	  //if we've recieved from our right, we go from soft boundary to the left into normal region
	  if(direction[j] == -1)
	    temp = soft_boundary_points_[j] + k + 1;
	  else
	    temp = grid_.grid_number_[i] - (soft_boundary_points_[j] + 1) - k;

	  //the value of the function
	  //the first point in the buffer was at the boundary, which is the deepest the
	  //we penetrate here. So we reverse
	  grid_.grid_[temp] = rec_buffer[buffer_size - k * (DIM+1)]
	  //the derivative
	  for(l = 0; l < DIM; l++) {
	    grid._grid_deriv_[temp * DIM + l] = rec_buffer[buffer_size - k * (DIM+1) - l]
	  }
	}
      }            
    }

    free(rec_buffer);
    free(send_buffer);
  }
  */

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
