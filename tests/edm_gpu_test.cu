#include "grid_gpu.cuh"
#include "edm_bias_gpu.cuh"
#include "gaussian_grid_gpu.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
//These must be declared here.
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE EDM_GPU

#define EPSILON 1e-10
#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define GRID_SRC std::string(STR(TEST_GRID_SRC))

#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <boost/test/unit_test.hpp>

#define TIMING_BOUND_edm_cpu_timer_1d 10000

using namespace boost;
using namespace EDM;
using namespace EDM_Kernels;

typedef chrono::duration<double> sec; // seconds, stored with a double




//Many of these test are the same as the serial ones, just to make sure we preserve behavior
BOOST_AUTO_TEST_CASE( grid_gpu_1d_sanity ){
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGridGPU<1> g (min, max, bin_spacing, periodic, 0, 0);
  DimmedGridGPU<1>* d_g;
  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 11);
  BOOST_REQUIRE_EQUAL(g.grid_size_, 11);
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<1>)));

  size_t array[] = {5};
  size_t temp[1];
  g.one2multi(g.multi2one(array), temp);
  BOOST_REQUIRE_EQUAL(array[0], temp[0]);

  for(int i = 0; i < 11; i++){
    g.grid_[i] = i;
  }
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<1>), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  double x[] = {3.5};
  BOOST_REQUIRE(g.in_grid(x));
  size_t index[1];
  g.get_index(x, index);
  BOOST_REQUIRE(index[0] - 3 < 0.000001);

  double* d_x;
  gpuErrchk(cudaMalloc(&d_x, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  double target[1] = {0.0};
  double* d_target;
  gpuErrchk(cudaMalloc((void**) &d_target, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_kernel<1><<<1,1>>>(d_x, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(pow(target[0] -3, 2) < 0.000001);

  //try to break it
  x[0] = 0;
  
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_kernel<1><<<1,1>>>(d_x, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(g.get_value(x) - target[0] < EPSILON);//require same behavior on host/dev

  x[0] = 10;
  
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_kernel<1><<<1,1>>>(d_x, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(g.get_value(x) - target[0] < EPSILON);//require same behavior on host/dev

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_target));
  gpuErrchk(cudaFree(d_x));
}//grid_gpu_1d_sanity

BOOST_AUTO_TEST_CASE( grid_gpu_3d_sanity ){
  double min[] = {-2, -5, -3};
  double max[] = {125, 63, 78};
  double bin_spacing[] = {1.27, 1.36, 0.643};
  int periodic[] = {0, 1, 1};
  DimmedGridGPU<3> g (min, max, bin_spacing, periodic, 0, 0);
  DimmedGridGPU<3>* d_g;
  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
  BOOST_REQUIRE_EQUAL(g.grid_number_[1], 50);
  BOOST_REQUIRE_EQUAL(g.grid_number_[2], 126);
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<3>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<3>), cudaMemcpyHostToDevice));

  size_t array[3];
//  size_t temp[3];
  size_t* d_array;
  size_t* d_temp;
  gpuErrchk(cudaMalloc((void**)&d_array, 3*sizeof(size_t)));
  gpuErrchk(cudaMalloc((void**)&d_temp, 3*sizeof(size_t)));

  for(int i = 0; i < g.grid_number_[0]; i++) {
    array[0] = i;
    for(int j = 0; j < g.grid_number_[1]; j++) {
      array[1] = j;
      for(int k = 0; k < g.grid_number_[2]; k++) {
	array[2] = k;
	  /* This passes but it's slow, so leaving out for now
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(d_array, array, 3*sizeof(size_t), cudaMemcpyHostToDevice));

	//g.one2multi(g.multi2one(array), temp);
	multi2one_kernel<3><<<1,1>>>( d_array, d_temp, d_g);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(array, d_array, 3*sizeof(size_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp, d_temp, 3*sizeof(size_t), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaDeviceSynchronize());
	BOOST_REQUIRE_EQUAL(array[0], temp[0]);
	BOOST_REQUIRE_EQUAL(array[1], temp[1]);
	BOOST_REQUIRE_EQUAL(array[2], temp[2]);
*/
	g.grid_[g.multi2one(array)] = g.multi2one(array);
      }
    }
  }
  
  double point[3];
  gpuErrchk(cudaDeviceSynchronize());
  for(int i = 0; i < g.grid_number_[0]; i++) {
    point[0] = i * g.dx_[0] + g.min_[0] + EPSILON;
    array[0] = i;
    for(int j = 0; j < g.grid_number_[1]; j++) {
      point[1] = j * g.dx_[1] + g.min_[1] + EPSILON;
      array[1] = j;
      for(int k = 0; k < g.grid_number_[2]; k++) {
	point[2] = k * g.dx_[2] + g.min_[2] + EPSILON;
	array[2] = k;
	BOOST_REQUIRE(pow(g.do_get_value(point) - g.grid_[g.multi2one(array)],2) < 0.0000001);
      }
    }
  }
  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_temp));
  gpuErrchk(cudaFree(d_array));
}//grid_gpu_3d_sanity

BOOST_AUTO_TEST_CASE( grid_gpu_1d_read ) {
  DimmedGridGPU<1> g(GRID_SRC + "/1.grid");
  BOOST_REQUIRE_EQUAL(g.min_[0], 0);
  BOOST_REQUIRE_EQUAL(g.max_[0], 2.5 + g.dx_[0]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
}

BOOST_AUTO_TEST_CASE( grid_gpu_3d_read ) {
  DimmedGridGPU<3> g(GRID_SRC + "/3.grid");//derivatives is true here
  DimmedGridGPU<3>* d_g;
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<3>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<3>), cudaMemcpyHostToDevice));

  BOOST_REQUIRE_EQUAL(g.min_[2], 0);
  BOOST_REQUIRE_EQUAL(g.max_[2], 2.5 + g.dx_[2]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[2], 11);
  double temp[] = {0.75, 0, 1.00};
  double* d_temp;
  double* d_target;
  double target[1] = {0.0};
  
  gpuErrchk(cudaMalloc((void**)&d_temp, 3*sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &d_target, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_temp, temp, 3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  
  get_value_kernel<3><<<1,1>>>(d_temp, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(target[0] - 1.260095, 2) < EPSILON);

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_temp));
  gpuErrchk(cudaFree(d_target));
}//grid_gpu_3d_read

BOOST_AUTO_TEST_CASE( gpu_derivative_direction ) {
  DimmedGridGPU<3> g(GRID_SRC + "/3.grid", 1);
  DimmedGridGPU<3>* d_g;
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<3>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<3>), cudaMemcpyHostToDevice));
  double temp[] = {0.75, 0, 1.00};
  double temp2[] = {0.76, 0, 1.00};
  double target[]={0.0};
  double target2[]={0.0};
  double* d_temp;
  double* d_temp2;
  double* d_target;
  double* d_target2;
  gpuErrchk(cudaMalloc((void**)&d_temp, 3*sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &d_target, sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&d_temp2, 3*sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &d_target2, sizeof(double)));
  
  gpuErrchk(cudaMemcpy(d_temp, temp, 3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_temp2, temp2, 3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target2, target2, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  
  get_value_kernel<3><<<1,1>>>(d_temp, d_target, d_g);
  get_value_kernel<3><<<1,1>>>(d_temp2, d_target2, d_g);
  
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(temp, d_temp, 3*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(temp2, d_temp2, 3*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target2, d_target2, sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(target2[0] > target[0]);
  
  temp2[0] = 0.75;
  temp2[2] = 0.99;
  target[0] = 0.0;
  target2[0] = 0.0;

  gpuErrchk(cudaMemcpy(d_temp, temp, 3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_temp2, temp2, 3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target2, target2, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());

  get_value_kernel<3><<<1,1>>>(d_temp, d_target, d_g);
  get_value_kernel<3><<<1,1>>>(d_temp2, d_target2, d_g);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(temp, d_temp, 3*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(temp2, d_temp2, 3*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target2, d_target2, sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(target2[0] < target[0]);
  gpuErrchk(cudaFree(d_temp));
  gpuErrchk(cudaFree(d_target));
  gpuErrchk(cudaFree(d_temp2));
  gpuErrchk(cudaFree(d_target2));
  gpuErrchk(cudaFree(d_g));
}//gpu_derivative_direction

BOOST_AUTO_TEST_CASE( grid_gpu_read_write_consistency ) {
  size_t i, j;
  std::string input;
  std::string output;
  for(i = 1; i <= 3; i++) {
    std::stringstream filename;
    filename << i << ".grid";
    input = GRID_SRC + "/" + filename.str();
    output = filename.str() + ".test";
    Grid* g;
    switch(i) {
    case 1:
      g = new DimmedGridGPU<1>(input);
      break;
    case 2:
      g = new DimmedGridGPU<2>(input);
      break;
    case 3:
      g = new DimmedGridGPU<3>(input);
      break;
    }
    g->write(output);
    //grab the grid for comparison
    size_t ref_length = g->get_grid_size();
    double ref_grid[ref_length];
    for(j = 0; j < ref_length; j++)
      ref_grid[j] = g->get_grid()[j];
    //re-read
    gpuErrchk(cudaDeviceSynchronize());
    g->read(output);
    //now compare
    BOOST_REQUIRE_EQUAL(g->get_grid_size(), ref_length);

    for(j = 0; j < ref_length; j++)
      BOOST_REQUIRE(pow(ref_grid[j] - g->get_grid()[j], 2) < EPSILON);

  }
}//grid_gpu_read_write_consistency

BOOST_AUTO_TEST_CASE( gpu_interpolation_1d ) {
  gpuErrchk(cudaDeviceReset());
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGridGPU<1> g (min, max, bin_spacing, periodic, 1, 1);
  
  for(int i = 0; i < 11; i++) {
    g.grid_[i] = log(i);
    g.grid_deriv_[i] = 1. / i;
  }

  DimmedGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<3>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<1>), cudaMemcpyHostToDevice));

  double array[] = {5.3};
  double der[1];
  double fhat[1] = {0.0};//g.get_value_deriv(array,der);
  double* d_array;
  double* d_der;
  double* d_fhat;
  gpuErrchk(cudaMalloc((void**)&d_array, sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&d_fhat, sizeof(double)));

  gpuErrchk(cudaMemcpy(d_array, array, sizeof(double), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_der, der, sizeof(double), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_fhat, fhat, sizeof(double), cudaMemcpyHostToDevice ));

  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost ));

  
  //make sure it's at least in the ballpark
  BOOST_REQUIRE(fhat[0] > log(5) && fhat[0] < log(6));
  BOOST_REQUIRE(der[0] < 1. / 5 && der[0] > 1. / 6.);

  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat[0] - log(5.3), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0]- 1. / 5.3, 2) < 0.1);

  //try edge cases
  array[0] = 5.0;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(double), cudaMemcpyHostToDevice ));
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost ));

  array[0] = 5.5;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(double), cudaMemcpyHostToDevice ));
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost ));

  array[0] = 0.0;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(double), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_der, der, sizeof(double), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_fhat, fhat, sizeof(double), cudaMemcpyHostToDevice ));
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost ));

  array[0] = 10.0;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(double), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(array, d_array, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost ));

  gpuErrchk(cudaFree(d_array));
  gpuErrchk(cudaFree(d_der));
  gpuErrchk(cudaFree(d_fhat));
  gpuErrchk(cudaFree(d_g));
}//gpu_interpolation_1d

BOOST_AUTO_TEST_CASE( gpu_interp_1d_periodic ) {
  double min[] = {-M_PI};
  double max[] = {M_PI};
  double bin_spacing[] = {M_PI / 100};
  int periodic[] = {1};
  DimmedGridGPU<1> g (min, max, bin_spacing, periodic, 1, 1);

  for(int i = 0; i < g.grid_number_[0]; i++) {
    g.grid_[i] = sin(g.min_[0] + i * g.dx_[0]);
    g.grid_deriv_[i] = cos(g.min_[0] + i * g.dx_[0]);
  }
  DimmedGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<1>), cudaMemcpyHostToDevice));


  double array[] = {M_PI / 4};
  double der[1];
  double fhat[1] = {0.0};//g.get_value_deriv(array,der);

  double* d_array;
  double* d_der;
  double* d_fhat;
  gpuErrchk(cudaMalloc((void**)&d_array, sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&d_fhat, sizeof(double)));

  gpuErrchk(cudaMemcpy((void**)d_array, array, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy((void**)d_fhat, fhat, sizeof(double), cudaMemcpyHostToDevice));

  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);
  
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));
  
  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat[0] - sin(array[0]), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - cos(array[0]), 2) < 0.1);

  //test periodic
  array[0] = 5 * M_PI / 4;
  gpuErrchk(cudaMemcpy((void**)d_array, array, sizeof(double), cudaMemcpyHostToDevice));
  get_value_deriv_kernel<1><<<1,1,32>>>(d_array, d_der, d_fhat, d_g);
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));
  
  //fhat = g.get_value_deriv(array,der);

  g.write("grid.test");
  
  BOOST_REQUIRE(pow(fhat[0] - sin(array[0]), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - cos(array[0]), 2) < 0.1);

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_array));
  gpuErrchk(cudaFree(d_der));
  gpuErrchk(cudaFree(d_fhat));
}//gpu_interp_1d_periodic

BOOST_AUTO_TEST_CASE( gpu_boundary_remap_wrap) {
  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {0, 0};
  double max[] = {10, 5};
  double bin_spacing[] = {1, 1};
  int periodic[] = {1, 0};
  double sigma[] = {0.1, 0.1};
  DimmedGaussGridGPU<2> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<2>* d_g;
  double* d_min;
  double* d_max;
  int* d_periodic;
  
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<2>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<2>), cudaMemcpyHostToDevice));
  
  max[1] = 10;
  periodic[1] = 1;

  gpuErrchk(cudaMalloc((void**)&d_min, 2*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_min, min, 2*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_max, 2*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_max, max, 2*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_periodic, 2*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, 2*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  set_boundary_kernel<2><<<1,1>>>(d_min, d_max, d_periodic, d_g);
  gpuErrchk(cudaDeviceSynchronize());

  g.set_boundary(min, max, periodic);

  double test_point[2] = {0.0,1.0}; //should not remap
  double* d_test_point;
  
  gpuErrchk(cudaMalloc((void**)&d_test_point, 2*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - 0, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[0] = -1;//on grid, at 9
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[1] = 6;//closest point is 6
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 6, 2) < 0.1);

  test_point[1] = 11;//actually in grid at 1
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[1] = 9; //closest point is -1
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - -1, 2) < 0.1);

  test_point[1] = -1; //closest point is -1
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - -1, 2) < 0.1);

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_min));
  gpuErrchk(cudaFree(d_max));
  gpuErrchk(cudaFree(d_periodic));
  gpuErrchk(cudaFree(d_test_point));

}//gpu_boundary_remap_wrap

BOOST_AUTO_TEST_CASE( gpu_boundary_remap_wrap_2) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {-2};
  double max[] = {7};
  double bin_spacing[] = {0.1};
  int periodic[] = {0};
  double sigma[] = {0.1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  min[0] = 0;
  double* d_min;
  max[0] = 10;
  double* d_max;
  periodic[0] = 1;
  int* d_periodic;
  
  gpuErrchk(cudaMalloc((void**)&d_min, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_min, min, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_max, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_max, max, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_periodic, sizeof(int)));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());

  set_boundary_kernel<1><<<1,1>>>(d_min, d_max, d_periodic, d_g);
  

  double test_point[] = {0}; //should not remap
  double* d_test_point;
  gpuErrchk(cudaMalloc((void**)&d_test_point, sizeof(double)));
  
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 0, 2) < 0.1);

  test_point[0] = -1;//shoul not remap
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - -1, 2) < 0.1);

  test_point[0] = 9;//should remap
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - -1, 2) < 0.1);

  test_point[0] = 6;//should not remap
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(double), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 6, 2) < 0.1);

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_min));
  gpuErrchk(cudaFree(d_max));
  gpuErrchk(cudaFree(d_periodic));
  gpuErrchk(cudaFree(d_test_point));
 
}//gpu_boundary_remap_wrap_2

BOOST_AUTO_TEST_CASE( gpu_boundary_remap_wrap_3) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {-2};
  double max[] = {7};
  double bin_spacing[] = {0.1};
  int periodic[] = {0};
  double sigma[] = {0.1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  min[0] = 0;
  max[0] = 10;
  periodic[0] = 1;
  g.set_boundary(min, max, periodic);//test if we can set bounds first, then copy

  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  double point[] = {0.01};
  double* d_point;
  gpuErrchk(cudaMalloc((void**)&d_point, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(double), cudaMemcpyHostToDevice));

//THE MOMENT WE'VE ALL BEEN WAITING FOR!!
  
//  g.add_value(point,1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_point, 1.0, d_g);

  double* d_target;
  gpuErrchk(cudaMalloc((void**)&d_target, sizeof(double)));//for checking the value
  double der[1];
  point[0] = 0;
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(double), cudaMemcpyHostToDevice));
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));//for filling with deriv
  get_value_deriv_kernel<1><<<1,1>>>(d_point, d_der, d_target, &(d_g->grid_));//gross

  gpuErrchk(cudaMemcpy(&der, d_der, sizeof(double), cudaMemcpyDeviceToHost));
  

//  g.get_value_deriv(point, der);
  BOOST_REQUIRE(fabs(der[0]) > 0.1);
}//gpu_boundary_remap_wrap_3

BOOST_AUTO_TEST_CASE( gpu_boundary_remap_nowrap_1) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {-2};
  double max[] = {7};
  double bin_spacing[] = {0.1};
  int periodic[] = {0};
  double sigma[] = {0.1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  min[0] = 0;
  max[0] = 10;
  periodic[0] = 0;
  g.set_boundary(min, max, periodic);
  
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));


  double point[] = {-0.01};
  double* d_point;
  gpuErrchk(cudaMalloc((void**)&d_point, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(double), cudaMemcpyHostToDevice));
//  g.add_value(point,1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_point, 1.0, d_g);

//  double der[1];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  point[0] = 0;
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(double), cudaMemcpyHostToDevice));
  double* d_target;
  gpuErrchk(cudaMalloc((void**)&d_target, sizeof(double)));
//  g.get_value_deriv(point, der);
  get_value_deriv_kernel<1><<<1,1>>>(d_point, d_der, d_target, &(d_g->grid_));//gross
  gpuErrchk(cudaMemcpy(point, d_point, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(fabs(point[0]) < EPSILON);
}//gpu_boundary_remap_nowrap_1

BOOST_AUTO_TEST_CASE( gpu_interp_3d_mixed ) {
  double min[] = {-M_PI, -M_PI, 0};
  double max[] = {M_PI, M_PI, 10};
  double bin_spacing[] = {M_PI / 100, M_PI / 100, 1};
  int periodic[] = {1, 1, 0};
  DimmedGridGPU<3> g (min, max, bin_spacing, periodic, 1, 0);
  size_t index = 0;
  double x,y,z;
  
  for(int i = 0; i < g.grid_number_[2]; i++) {
    for(int j = 0; j < g.grid_number_[1]; j++) {
      for(int k = 0; k < g.grid_number_[0]; k++) {
	x = g.min_[0] + k * g.dx_[0];
	y = g.min_[1] + j * g.dx_[1];
	z = g.min_[2] + i * g.dx_[2];
	g.grid_[index] = cos(x) * sin(y) * z;
	g.grid_deriv_[index * 3 + 0] = -sin(x) * sin(y) * z;
	g.grid_deriv_[index * 3 + 1] = cos(x) * cos(y) * z;
	g.grid_deriv_[index * 3 + 2] = cos(x) * sin(y);
	index++;
      }
    }
  }
  DimmedGridGPU<3>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGridGPU<3>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<3>), cudaMemcpyHostToDevice));


  double array[] = {-10.75 * M_PI / 2, 8.43 * M_PI / 2, 3.5};
  double* d_array;
  gpuErrchk(cudaMalloc((void**)&d_array, 3*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_array, array, 3*sizeof(double), cudaMemcpyHostToDevice));
  double der[3];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, 3*sizeof(double)));
  double* d_fhat;
  gpuErrchk(cudaMalloc((void**)&d_fhat, sizeof(double)));
  get_value_deriv_kernel<3><<<1,1>>>(d_array, d_der, d_fhat, d_g);
  double fhat[1];
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(der, d_der, 3*sizeof(double), cudaMemcpyDeviceToHost));
//  double fhat = g.get_value_deriv(array,der);
  double f = cos(array[0]) * sin(array[1]) * array[2];
  double true_der[] = {-sin(array[0]) * sin(array[1]) * array[2],
		       cos(array[0]) * cos(array[1]) * array[2],
		       cos(array[0]) * sin(array[1])};
  
  BOOST_REQUIRE(pow(f- fhat[0], 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - true_der[0], 2) < 0.1);
  BOOST_REQUIRE(pow(der[1] - true_der[1], 2) < 0.1);
  BOOST_REQUIRE(pow(der[2] - true_der[2], 2) < 0.1);

}//gpu_interp_3d_mixed

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_add_check ) {
  double min[] = {-10};
  double max[] = {10};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 0, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
  //add 1 gaussian
  double x[] = {0};
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
//  g.add_value(x, 1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.0, d_g);

  //now check a few points
  double* d_target;
  gpuErrchk(cudaMalloc((void**)&d_target, sizeof(double)));
  get_value_kernel<1><<<1,1>>>(d_x, d_target, &(d_g->grid_));
  double target[1];
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(pow(target[0] - 1 / sqrt(2 * M_PI), 2) < EPSILON);

  int i;
  double der[1];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  for( i = -6; i < 7; i++) {
    x[0] = i;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
//    value = g.get_value_deriv(x, der);
    get_value_deriv_kernel<1><<<1, 1>>>(d_x, d_der, d_target, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));
    BOOST_REQUIRE(pow(target[0] - exp(-x[0]*x[0]/2.) / sqrt(2*M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-x[0] *exp(-x[0]*x[0]/2.)) / sqrt(2*M_PI), 2) < 0.01);
  }
 
}//gpu_gauss_grid_add_check

BOOST_AUTO_TEST_CASE( gpu_gauss_pbc_check ) {
  double min[] = {2};
  double max[] = {10};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 0, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add 1 gaussian
  double x[] = {2};
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
//  g.add_value(x, 1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.0, d_g);
  
  int i;
  double der[1];
  double value[1];
  double* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(double)));
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  double dx;//delta-x, not device_x
  for( i = -6; i < 7; i++) {
    x[0] = i;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    dx = x[0] - 2;
    dx -= round(dx / (min[0] - max[0])) * (min[0] - max[0]);
//    value = g.get_value_deriv(x, der);
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_value, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(value, d_value, sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "x = " << x[0]
	      << " dx = " << dx 
	      << "(" 
	      << " value = " << value[0]
	      << " (" << exp(-dx*dx/2.) / sqrt(2 * M_PI) << ")" 
	      << std::endl;

    BOOST_REQUIRE(pow(value[0] - exp(-dx*dx/2.) / sqrt(2 * M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-dx *exp(-dx*dx/2.)) / sqrt(2 * M_PI), 2) < 0.01);
  }
 
}//gpu_gauss_pbc_check

BOOST_AUTO_TEST_CASE( gpu_gauss_subdivided_pbc_check ) {
  double min[] = {2};
  double max[] = {4};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  double gauss_loc[] = {11};
  double* d_gauss_loc;
  gpuErrchk(cudaMalloc((void**)&d_gauss_loc, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_gauss_loc, gauss_loc, sizeof(double), cudaMemcpyHostToDevice));
  double x[1];
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 0, sigma);
  periodic[0] = 1;
  max[0] = 10;
  g.set_boundary(min, max, periodic);

  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add 1 gaussian
//  g.add_value(gauss_loc, 1); //added at equivalent to 1
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_gauss_loc, 1.0, d_g);

  int i;
  double der[1];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  double value[1];
  double* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(double)));
  double dx;
  double* d_x;//device copy of x
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  for( i = 2; i < 4; i++) {
    x[0] = i;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    dx = x[0] - gauss_loc[0];
    dx  -= round(dx / (min[0] - max[0])) * (min[0] - max[0]);
//    value = g.get_value_deriv(x, der);
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_value, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(value, d_value, sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));

    BOOST_REQUIRE(pow(value[0] - exp(-dx*dx/2.) / sqrt(2 * M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-dx *exp(-dx*dx/2.)) / sqrt(2 * M_PI), 2) < 0.01);
  }
 
}//gpu_gauss_subdivided_pbc_check

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_integral_test ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {1.2};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
  
  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  double offsets = 1. / N;
  double g_integral[g.minisize_total_];
  double* d_g_integral;
  gpuErrchk(cudaMalloc((void**)&d_g_integral, g.minisize_total_*sizeof(double)));

  double g_integral_total = 0;
  //generate a random number but use sequential grid point offsets
  int j;
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    add_value_integral_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.5, d_g_integral, d_g);
    gpuErrchk(cudaMemcpy(g_integral, d_g_integral, g.minisize_total_*sizeof(double), cudaMemcpyDeviceToHost));
    for(j = 0; j < g.minisize_total_; j++){
      g_integral_total += g_integral[j];
    }
  }
  
  //now we integrate the grid
  double area = 0;
  double value[1] = {0};
  double dx = 0.1;
  double* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(double)));
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(value, d_value, sizeof(double), cudaMemcpyDeviceToHost));
    area += value[0] * dx;
  }

  //Make sure the integrated area is correct
  //unnormalized, so a little height scaling is necessary
  //  std::cout << area << " " << N * 1.5 << std::endl;
  BOOST_REQUIRE(pow(area - N * 1.5, 2) < 1);

  //now make sure that add_value returned the correct answers as well
  BOOST_REQUIRE(pow(area - g_integral_total, 2) < 0.1);
}//gpu_gauss_grid_integral_test

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_derivative_test ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {1.2};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  
  double offsets = 1. / N;
  
  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.5, d_g);
  }

  //now we calculate finite differences on the grid
  double vlast, vlastlast, v, approx_der;
  double* d_v;
  gpuErrchk(cudaMalloc((void**)&d_v, sizeof(double)));

  double der[1];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  double der_last;
  double dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_v, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(&v, d_v, sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));
    if(i > 1) {
      approx_der = (v - vlastlast) / (2*dx);
      BOOST_REQUIRE(pow(approx_der - der_last, 2) < 0.01);
    }
    vlastlast = vlast;
    vlast = v;

    der_last = der[0];
  }

}//gpu_gauss_grid_derivative_test

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_derivative_test_mcgdp_1 ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {1.2};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  double offsets = 1. / N;

  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.5, d_g);
  }

  //now we calculate finite differences on the grid
  double vlast, vlastlast, v, approx_der;
  double* d_v;
  gpuErrchk(cudaMalloc((void**)&d_v, sizeof(double)));


  double der[1];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));
  double der_last;
  double dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_v, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(&v, d_v, sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));

    if(i > 1) {
      approx_der = (v - vlastlast) / (2*dx);
      BOOST_REQUIRE(pow(approx_der - der_last, 2) < 0.001);
    } else {
      BOOST_REQUIRE(pow(der[0], 2) < 0.001);
    }
    vlastlast = vlast;
    vlast = v;

    der_last = der[0];
  }
  
  approx_der = (vlast - vlastlast) / dx;
  BOOST_REQUIRE(pow(approx_der - der_last, 2) < 0.1);
  BOOST_REQUIRE(pow(der_last, 2) < 0.01);

}//gpu_gauss_grid_derivative_test_mcgdp_1

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_interp_test_mcgdp_1D ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {10.0};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  periodic[0]  = 0;
  min[0] = -50;
  max[0] = 50;
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
  double* d_min;
  double* d_max;
  int* d_periodic;
  gpuErrchk(cudaMalloc((void**)&d_min, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_min, min, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_max, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_max, max, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_periodic, sizeof(int)));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, sizeof(int), cudaMemcpyHostToDevice));
  set_boundary_kernel<1><<<1,1>>>(d_min, d_max, d_periodic, d_g);



  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  double der[1];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(double)));

  //generate a random number
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
    add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.0, d_g);
  }

  //Check if the boundaries were duplicated
  double value[1];
  double value2[1];
  double* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(double)));
  x[0] = 50.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value, d_value, sizeof(double), cudaMemcpyDeviceToHost));

  x[0] = 49.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value2, d_value, sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(value[0] - value2[0] ,2) < 2*(EPSILON));//This one cuts it too close...?

  x[0] = 150.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value, d_value, sizeof(double), cudaMemcpyDeviceToHost));

  x[0] = 151.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value2, d_value, sizeof(double), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(value[0] - value2[0] ,2) < EPSILON);

  x[0] = 50.0;

  //boundaries should be 0, even with interpolation
//  g.get_value_deriv(x,der);
  double* d_dummy;
  gpuErrchk(cudaMalloc((void**)&d_dummy, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);

  //check other side
  x[0] = -50.1;
  x[0] = -50.0;
  gpuErrchk(cudaMalloc((void**)&d_dummy, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);

}//gpu_gauss_grid_interp_test_mcgdp_1D

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_interp_test_mcgdp_3D ) {
  double min[] = {-10, -10, -10};
  double* d_min;
  gpuErrchk(cudaMalloc((void**)&d_min, 3*sizeof(double)));
  double max[] = {10, 10, 10};
  double* d_max;
  gpuErrchk(cudaMalloc((void**)&d_max, 3*sizeof(double)));
  double sigma[] = {3.0, 3.0, 3.0};
  double bin_spacing[] = {0.9, 1.1, 1.4};
  int periodic[] = {1, 1, 1};
  int* d_periodic;
  gpuErrchk(cudaMalloc((void**)&d_periodic, 3*sizeof(int)));
  DimmedGaussGridGPU<3> g (min, max, bin_spacing, periodic, 1, sigma);
  periodic[0]  = periodic[1] = periodic[2] = 0;
  min[0]  = min[1] = min[2] = -5;
  max[0]  = max[1] = max[2] = 5;

  gpuErrchk(cudaMemcpy(d_min, min, 3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_max, max, 3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, 3*sizeof(int), cudaMemcpyHostToDevice));
  
//  g.set_boundary(min, max, periodic);


  DimmedGaussGridGPU<3>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<3>)));

  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<3>), cudaMemcpyHostToDevice));

  set_boundary_kernel<3><<<1,1>>>(d_min, d_max, d_periodic, d_g);

  //add N gaussian
  int N = 20;
  int i;
  double x[3];
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, 3*sizeof(double)));
  double der[3];
  double* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, 3*sizeof(double)));
  double v;
  double* d_v;
  gpuErrchk(cudaMalloc((void**)&d_v, 3*sizeof(double)));

  //generate a random number
  for(i = 0; i < N; i++) {
    x[0] = rand() % 20 - 10;
    x[1] = rand() % 20 - 10;
    x[2] = rand() % 20 - 10;
    gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(double), cudaMemcpyHostToDevice));
    add_value_kernel<3><<<1, g.minisize_total_>>>(d_x, 5.0, d_g);
  }

  //Check if the boundaries were duplicated
  double v2;
  x[0] = x[2] = 50.1;
  x[1] = 5.0;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v2, d_v, sizeof(double), cudaMemcpyDeviceToHost));
  x[0] = x[1] = 50.0;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v, d_v, sizeof(double), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(v - v2, 2) < EPSILON);

  //boundaries should be 0, even with interpolation
  double* d_dummy;
  gpuErrchk(cudaMalloc((void**)&d_dummy, sizeof(double)));
//  g.get_value_deriv(x,der);
  get_value_deriv_kernel<3><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(der, d_der, 3*sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < 0.001);

  //check another location
  x[0] = -5.1;
  x[2] = 5.1;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v, d_v, sizeof(double), cudaMemcpyDeviceToHost));
  x[0] = x[2] = -5.0;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(double), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v2, d_v, sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(pow(v - v2, 2) < 0.001);
  get_value_deriv_kernel<3><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(der, d_der, 3*sizeof(double), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);

}//gpu_gauss_grid_interp_test_mcgdp_3D

/*BOOST_AUTO_TEST_CASE( gpu_gauss_grid_integral_regression_1 ) {
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {0.009765625};
  double sigma[] = {0.1};
  int periodic[] = {1};
  GaussGrid* g  = make_gauss_grid_gpu(1, min, max, bin_spacing, periodic, 1, sigma);
  periodic[0] = 1;
  g->set_boundary(min, max, periodic);

  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add gaussian that was failing
  double x[] = {-3.91944};
  double* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice));
  double bias_added[32];
  double* d_bias_added;
  gpuErrchk(cudaMalloc((void**)&d_bias_added, 32*sizeof(double)));
  add_value_integral_kernel<1><<<1, 32>>>(d_x, 1.0, d_bias_added, d_g);
  gpuErrchk(cudaMemcpy(bias_added, d_bias_added, 32*sizeof(double), cudaMemcpyDeviceToHost));
  double bias_added_tot = 0.0;
  for(int i = 0; i < 32; i++){
    bias_added_tot += bias_added[i];
  }
//  double bias_added = g->add_value(x, 1.0);

  //unnormalized, so a little height scaling is necessary
  //std::cout << bias_added /  (sqrt(2 * M_PI) * sigma[0]) << " " << h << std::endl;
  BOOST_REQUIRE(pow(bias_added_tot - 1.0, 2) < 0.1);

  delete g;
}//gpu_gauss_grid_integral_regression_1
*/


//This test will simply run several thousand timesteps and time how long it takes.
BOOST_AUTO_TEST_CASE( edm_cpu_timer_1d ){
  
  double min[] = {-10};
  double max[] = {10};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  double x[1] = {0};
  unsigned int n_hills = 5000;
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 0, sigma);
  
  //now just do a generic loop, adding 10k gaussians, and time it
  boost::timer::auto_cpu_timer t;
  for( unsigned int i = 0; i < n_hills; i++){
    int rand_num = rand() % 20 - 10;
    x[0] = rand_num;
    g.add_value(x,1);
  }
  t.stop();
  sec seconds = chrono::nanoseconds(t.elapsed().user);
  
  BOOST_REQUIRE(seconds.count() < TIMING_BOUND_edm_cpu_timer_1d);
}

//BOOST_AUTO_TEST_SUITE_END()

