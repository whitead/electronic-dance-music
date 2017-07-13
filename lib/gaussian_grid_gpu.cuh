#ifndef GPU_GAUSS_GRID_CH_
#define GPU_GAUSS_GRID_CH_

#include "gaussian_grid.h"
#include "grid_gpu.cuh"
#include "edm.h"
#include <string>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#ifndef GAUSS_SUPPORT
#define GAUSS_SUPPORT 8.0 // sigma^2 considered for gaussian
#endif
#ifndef BC_TABLE_SIZE
#define BC_TABLE_SIZE 65536 //boundary correction function look up size
#endif
#ifndef BC_MAR
#define BC_MAR 2.0
#endif
#ifndef BC_CORRECTION
#define BC_CORRECTION
#endif


namespace EDM{

/**
 * This class uses a compositional ("has-a") relationship with Grid. This is the 
 * interface and DimmedGaussGrid has the dimension template parameter
 **/
  class GaussGridGPU : public GaussGrid {
    /**
     * Retruns the integrated amount of bias added to the system
     **/
  public:
    virtual ~GaussGridGPU() {};
    double add_value(const double* x, double height) = 0;

//  __device__ virtual double add_hills_gpu(const double* buffer, const size_t hill_number, char hill_type, double *grid_);
  };

  template< int DIM>
  class DimmedGaussGridGPU : public DimmedGaussGrid<DIM>{
    /** A class for treating grids that have gaussians on it 
     *
     *
     **/
  public:
    DimmedGaussGridGPU(const double* min, 
		       const double* max, 
		       const double* bin_spacing, 
		       const int* b_periodic, 
		       int b_interpolate,
		       const double* sigma) : DimmedGaussGrid<DIM>(min, max, bin_spacing, b_periodic, b_interpolate, sigma), grid_(min, max, bin_spacing, b_periodic, 1, b_interpolate) {
      
      size_t i;
      for(i = 0; i < DIM; i++) {
	sigma_[i] = sigma[i] * sqrt(2.);
      }
    
      set_boundary(min, max, b_periodic);
      update_minigrid();
    
    }

    /**
     * Rebuild from a file. Files don't store sigma, so it must be set again.
     **/
    DimmedGaussGridGPU(const std::string& filename, const double* sigma) : DimmedGaussGrid<DIM>( filename, sigma), grid_(filename) {}
  
    ~DimmedGaussGridGPU() {
      //nothing
    }

    HOST_DEV void do_remap(double x[DIM]) const {
#ifdef __CUDACC__
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
      return;
#else
      remap(x);
#endif //CUDACC
      
    }//do_remap


    HOST_DEV void do_set_boundary(const double* min, const double* max, const int* b_periodic) {
#ifdef __CUDACC__
      //do the set_boundary on GPU
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
      return;
#else
      set_boundary(min, max, b_periodic);
#endif //CUDACC

    }//do_set_boundary
    
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

    HOST_DEV void do_duplicate_boundary(){
#ifdef __CUDACC__
      size_t i,j,k,l;
      size_t index_outter[DIM], index_bound[DIM];
      size_t min_i[DIM], max_i[DIM];
      int b_flag;
      printf("running do_duplicate_boundary from GPU!\n");

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
      size_t offset_size = 1;//pow(4,DIM);
      i = 0;
      while(i < DIM){
	offset_size*=4;
	i++;
      }
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
      }//for i in (0, offset_size)
    
      
      return;
#else
      duplicate_boundary();
#endif//CUDACC
    }//do_duplicate_boundary

    HOST_DEV double do_add_value(const double* x0, double height) {
#ifdef __CUDACC__ //device version
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


      do_remap(x); //attempt to remap to be close or in grid

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
      i = threadIdx.x;
      if( i < minisize_total_) {
      
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
	if(!b_flag){


      
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
	  printf("about to run do_duplicate_boundary from within do_add_value on GPU!\n");
	  do_duplicate_boundary();
	  b_dirty_bounds = 0;
	}

      }//if(i < minisize_total)
      return bias_added;
    
      //return(0);
#else
      return(add_value(x0, height));

#endif//CUDACC
    }//do_add_value



    //need to tell compiler where to find these since we have a derived templated class

    using DimmedGaussGrid<DIM>::minisize_;
    using DimmedGaussGrid<DIM>::minisize_total_;
    using DimmedGaussGrid<DIM>::multi_write;
    using DimmedGaussGrid<DIM>::get_volume;
    using DimmedGaussGrid<DIM>::in_bounds;
    using DimmedGaussGrid<DIM>::sigma_;
    using DimmedGaussGrid<DIM>::update_minigrid;
    using DimmedGaussGrid<DIM>::duplicate_boundary;
    using DimmedGaussGrid<DIM>::b_dirty_bounds;
    using DimmedGaussGrid<DIM>::boundary_min_;
    using DimmedGaussGrid<DIM>::boundary_max_;
    using DimmedGaussGrid<DIM>::b_periodic_boundary_;
    using DimmedGaussGrid<DIM>::bc_denom_table_;
    using DimmedGaussGrid<DIM>::bc_denom_deriv_table_;
    using DimmedGaussGrid<DIM>::add_value;

    DimmedGridGPU<DIM> grid_;//the underlying grid is a GPU-able grid
  };//DimmedGaussGridGPU class
  
/**
 * Used to avoid template constructors
 **/
  GaussGrid* make_gauss_grid_gpu( int dim, 
				  const double* min, 
				  const double* max, 
				  const double* bin_spacing, 
				  const int* b_periodic, 
				  int b_interpolate,
				  const double* sigma);

/**
 * Used to avoid template constructors
 **/
  GaussGrid* read_gauss_grid_gpu( int dim, const std::string& filename, const double* sigma);

}

namespace EDM_Kernels{
  //the kernels exist for testing purposes.
  using namespace EDM;
  
  /*
   * Kernel wrapper for do_set_boundary() on the GPU. Takes in an instance of DimmedGaussGridGPU
   */
  template <int DIM>
  __global__ void set_boundary_kernel(const double* min, const double* max,
				      const int* periodic, DimmedGaussGridGPU<DIM>* g){
    g->do_set_boundary(min, max, periodic);
    return;
  }

  /*
   * Kernel wrapper for do_remap() on the GPU. Takes in an instance of DimmedGaussGridGPU
   */
  template <int DIM>
  __global__ void remap_kernel(double* point, DimmedGaussGridGPU<DIM>* g){
    g->do_remap(point);
    return;
  }

  /*
   * Kernel wrapper for do_add_value() on the GPU. Takes in a point, the hill height to add, 
   * and an instance of DimmedGaussGridGPU to do the adding.
   */
  template <int DIM>
  __global__ void add_value_kernel(const double* point, double height, DimmedGaussGridGPU<DIM>* g){
    g->do_add_value(point, height);
    return;
  }

  
}

#endif //GPU_GAUSS_GRID_CH_
