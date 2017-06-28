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
      printf("doing do_remap on the GPU!\n");
      printf("test: can I access stuff from grid_? grid_.b_periodic_[0] is %d\n", grid_.b_periodic_[0]);

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
      
    }


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



    //need to tell compiler where to find these since we have a derived templated class

    using DimmedGaussGrid<DIM>::multi_write;
    using DimmedGaussGrid<DIM>::get_volume;
    using DimmedGaussGrid<DIM>::in_bounds;
    using DimmedGaussGrid<DIM>::sigma_;
    using DimmedGaussGrid<DIM>::update_minigrid;
    using DimmedGaussGrid<DIM>::b_dirty_bounds;
    using DimmedGaussGrid<DIM>::boundary_min_;
    using DimmedGaussGrid<DIM>::boundary_max_;
    using DimmedGaussGrid<DIM>::b_periodic_boundary_;
    using DimmedGaussGrid<DIM>::bc_denom_table_;
    using DimmedGaussGrid<DIM>::bc_denom_deriv_table_;

    DimmedGridGPU<DIM> grid_;//the underlying grid is a GPU-able grid
  };
  
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

  
}

#endif //GPU_GAUSS_GRID_CH_
