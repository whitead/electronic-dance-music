#include "grid_gpu.cuh"
#include "edm_bias_gpu.cuh"
#include "gaussian_grid_gpu.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
//These must be declared here.
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE EDM_GPU

#define EPSILON 1e-6
#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define GRID_SRC std::string(STR(TEST_GRID_SRC))
#define EDM_SRC std::string(STR(TEST_EDM_SRC))

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
  edm_data_t min[] = {0};
  edm_data_t max[] = {10};
  edm_data_t bin_spacing[] = {1};
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
  edm_data_t x[] = {3.5};
  BOOST_REQUIRE(g.in_grid(x));
  size_t index[1];
  g.get_index(x, index);
  BOOST_REQUIRE(index[0] - 3 < EPSILON);

  edm_data_t* d_x;
  gpuErrchk(cudaMalloc(&d_x, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  edm_data_t target[1] = {0.0};
  edm_data_t* d_target;
  gpuErrchk(cudaMalloc((void**) &d_target, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_kernel<1><<<1,1>>>(d_x, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(pow(target[0] -3, 2) < EPSILON);

  //try to break it
  x[0] = 0;
  
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_kernel<1><<<1,1>>>(d_x, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(g.get_value(x) - target[0] < EPSILON);//require same behavior on host/dev

  x[0] = 10;
  
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_kernel<1><<<1,1>>>(d_x, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(g.get_value(x) - target[0] < EPSILON);//require same behavior on host/dev

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_target));
  gpuErrchk(cudaFree(d_x));
}//grid_gpu_1d_sanity

BOOST_AUTO_TEST_CASE( grid_gpu_3d_sanity ){
  edm_data_t min[] = {-2, -5, -3};
  edm_data_t max[] = {125, 63, 78};
  edm_data_t bin_spacing[] = {1.27, 1.36, 0.643};
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
	//gpuErrchk(cudaDeviceSynchronize());
	BOOST_REQUIRE_EQUAL(array[0], temp[0]);
	BOOST_REQUIRE_EQUAL(array[1], temp[1]);
	BOOST_REQUIRE_EQUAL(array[2], temp[2]);
	  */
	    g.grid_[g.multi2one(array)] = g.multi2one(array);
      }
    }
  }
  
  edm_data_t point[3];
  edm_data_t denom, val;
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
	//printf("indices are i:%d j:%d k:%d\n", i, j, k);
	//printf("do_get_value gives %1.9f while g.grid_[g.multi2one(array)] gives %1.9f. \ng.multi2one(array) is %d\n", g.do_get_value(point),g.grid_[g.multi2one(array)],g.multi2one(array));
	//test
	val = g.do_get_value(point);
	denom = (val > EPSILON ? val : edm_data_t(1.0));
	BOOST_REQUIRE(((edm_data_t(g.do_get_value(point)) - edm_data_t(g.grid_[g.multi2one(array)]))/denom) < edm_data_t(EPSILON));
      }
    }
  }
  printf("for debugging\n");
  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_temp));
  gpuErrchk(cudaFree(d_array));
}//grid_gpu_3d_sanity

BOOST_AUTO_TEST_CASE( grid_gpu_1d_read ) {
  DimmedGridGPU<1> g(GRID_SRC + "/1.grid");
  BOOST_REQUIRE_EQUAL(g.min_[0], 0);
  BOOST_REQUIRE((g.max_[0] - (2.5 + g.dx_[0]))/g.max_[0] < EPSILON);
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
  edm_data_t temp[] = {0.75, 0, 1.00};
  edm_data_t* d_temp;
  edm_data_t* d_target;
  edm_data_t target[1] = {0.0};
  
  gpuErrchk(cudaMalloc((void**)&d_temp, 3*sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**) &d_target, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_temp, temp, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  
  get_value_kernel<3><<<1,1>>>(d_temp, d_target, d_g);
  gpuErrchk(cudaThreadSynchronize());
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

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
  edm_data_t temp[] = {0.75, 0, 1.00};
  edm_data_t temp2[] = {0.76, 0, 1.00};
  edm_data_t target[]={0.0};
  edm_data_t target2[]={0.0};
  edm_data_t* d_temp;
  edm_data_t* d_temp2;
  edm_data_t* d_target;
  edm_data_t* d_target2;
  gpuErrchk(cudaMalloc((void**)&d_temp, 3*sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**) &d_target, sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**)&d_temp2, 3*sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**) &d_target2, sizeof(edm_data_t)));
  
  gpuErrchk(cudaMemcpy(d_temp, temp, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_temp2, temp2, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target2, target2, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  
  get_value_kernel<3><<<1,1>>>(d_temp, d_target, d_g);
  get_value_kernel<3><<<1,1>>>(d_temp2, d_target2, d_g);
  
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(temp, d_temp, 3*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(temp2, d_temp2, 3*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target2, d_target2, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(target2[0] > target[0]);
  
  temp2[0] = 0.75;
  temp2[2] = 0.99;
  target[0] = 0.0;
  target2[0] = 0.0;

  gpuErrchk(cudaMemcpy(d_temp, temp, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target, target, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_temp2, temp2, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target2, target2, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());

  get_value_kernel<3><<<1,1>>>(d_temp, d_target, d_g);
  get_value_kernel<3><<<1,1>>>(d_temp2, d_target2, d_g);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(temp, d_temp, 3*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(temp2, d_temp2, 3*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target2, d_target2, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
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
    edm_data_t ref_grid[ref_length];
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
  edm_data_t min[] = {0};
  edm_data_t max[] = {10};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGridGPU<1> g (min, max, bin_spacing, periodic, 1, 1);
  
  for(int i = 0; i < 11; i++) {
    g.grid_[i] = log(i);
    g.grid_deriv_[i] = 1. / i;
  }

  DimmedGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<3>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<1>), cudaMemcpyHostToDevice));

  edm_data_t array[] = {5.3};
  edm_data_t der[1];
  edm_data_t fhat[1] = {0.0};//g.get_value_deriv(array,der);
  edm_data_t* d_array;
  edm_data_t* d_der;
  edm_data_t* d_fhat;
  gpuErrchk(cudaMalloc((void**)&d_array, sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**)&d_fhat, sizeof(edm_data_t)));

  gpuErrchk(cudaMemcpy(d_array, array, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_der, der, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_fhat, fhat, sizeof(edm_data_t), cudaMemcpyHostToDevice ));

  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));

  
  //make sure it's at least in the ballpark
  BOOST_REQUIRE(fhat[0] > log(5) && fhat[0] < log(6));
  BOOST_REQUIRE(der[0] < 1. / 5 && der[0] > 1. / 6.);

  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat[0] - log(5.3), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0]- 1. / 5.3, 2) < 0.1);

  //try edge cases
  array[0] = 5.0;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));

  array[0] = 5.5;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));

  array[0] = 0.0;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_der, der, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(d_fhat, fhat, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);

  gpuErrchk(cudaMemcpy(array, d_array, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));

  array[0] = 10.0;
  gpuErrchk(cudaMemcpy(d_array, array, sizeof(edm_data_t), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaDeviceSynchronize());
  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(array, d_array, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost ));

  gpuErrchk(cudaFree(d_array));
  gpuErrchk(cudaFree(d_der));
  gpuErrchk(cudaFree(d_fhat));
  gpuErrchk(cudaFree(d_g));
}//gpu_interpolation_1d

BOOST_AUTO_TEST_CASE( gpu_interp_1d_periodic ) {
  edm_data_t min[] = {-M_PI};
  edm_data_t max[] = {M_PI};
  edm_data_t bin_spacing[] = {M_PI / 100};
  int periodic[] = {1};
  DimmedGridGPU<1> g (min, max, bin_spacing, periodic, 1, 1);

  for(int i = 0; i < g.grid_number_[0]; i++) {
    g.grid_[i] = sin(g.min_[0] + i * g.dx_[0]);
    g.grid_deriv_[i] = cos(g.min_[0] + i * g.dx_[0]);
  }
  DimmedGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**) &d_g, sizeof(DimmedGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGridGPU<1>), cudaMemcpyHostToDevice));


  edm_data_t array[] = {M_PI / 4};
  edm_data_t der[1];
  edm_data_t fhat[1] = {0.0};//g.get_value_deriv(array,der);

  edm_data_t* d_array;
  edm_data_t* d_der;
  edm_data_t* d_fhat;
  gpuErrchk(cudaMalloc((void**)&d_array, sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  gpuErrchk(cudaMalloc((void**)&d_fhat, sizeof(edm_data_t)));

  gpuErrchk(cudaMemcpy((void**)d_array, array, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy((void**)d_fhat, fhat, sizeof(edm_data_t), cudaMemcpyHostToDevice));

  get_value_deriv_kernel<1><<<1,1>>>(d_array, d_der, d_fhat, d_g);
  
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat[0] - sin(array[0]), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - cos(array[0]), 2) < 0.1);

  //test periodic
  array[0] = 5 * M_PI / 4;
  gpuErrchk(cudaMemcpy((void**)d_array, array, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_deriv_kernel<1><<<1,1,32>>>(d_array, d_der, d_fhat, d_g);
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
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

  edm_data_t min[] = {0, 0};
  edm_data_t max[] = {10, 5};
  edm_data_t bin_spacing[] = {1, 1};
  int periodic[] = {1, 0};
  edm_data_t sigma[] = {0.1, 0.1};
  DimmedGaussGridGPU<2> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<2>* d_g;
  edm_data_t* d_min;
  edm_data_t* d_max;
  int* d_periodic;
  
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<2>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<2>), cudaMemcpyHostToDevice));
  
  max[1] = 10;
  periodic[1] = 1;

  gpuErrchk(cudaMalloc((void**)&d_min, 2*sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_min, min, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_max, 2*sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_max, max, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_periodic, 2*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, 2*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  set_boundary_kernel<2><<<1,1>>>(d_min, d_max, d_periodic, d_g);
  gpuErrchk(cudaDeviceSynchronize());

  g.set_boundary(min, max, periodic);

  edm_data_t test_point[2] = {0.0,1.0}; //should not remap
  edm_data_t* d_test_point;
  
  gpuErrchk(cudaMalloc((void**)&d_test_point, 2*sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - 0, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[0] = -1;//on grid, at 9
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[1] = 6;//closest point is 6
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 6, 2) < 0.1);

  test_point[1] = 11;//actually in grid at 1
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[1] = 9; //closest point is -1
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - -1, 2) < 0.1);

  test_point[1] = -1; //closest point is -1
  gpuErrchk(cudaMemcpy(d_test_point, test_point, 2*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<2><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, 2*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
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

  edm_data_t min[] = {-2};
  edm_data_t max[] = {7};
  edm_data_t bin_spacing[] = {0.1};
  int periodic[] = {0};
  edm_data_t sigma[] = {0.1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  min[0] = 0;
  edm_data_t* d_min;
  max[0] = 10;
  edm_data_t* d_max;
  periodic[0] = 1;
  int* d_periodic;
  
  gpuErrchk(cudaMalloc((void**)&d_min, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_min, min, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_max, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_max, max, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_periodic, sizeof(int)));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());

  set_boundary_kernel<1><<<1,1>>>(d_min, d_max, d_periodic, d_g);
  

  edm_data_t test_point[] = {0}; //should not remap
  edm_data_t* d_test_point;
  gpuErrchk(cudaMalloc((void**)&d_test_point, sizeof(edm_data_t)));
  
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(test_point[0] - 0, 2) < 0.1);

  test_point[0] = -1;//shoul not remap
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - -1, 2) < 0.1);

  test_point[0] = 9;//should remap
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(test_point[0] - -1, 2) < 0.1);

  test_point[0] = 6;//should not remap
  gpuErrchk(cudaMemcpy(d_test_point, test_point, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  remap_kernel<1><<<1,1>>>(d_test_point, d_g);
  gpuErrchk(cudaMemcpy(test_point, d_test_point, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
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

  edm_data_t min[] = {-2};
  edm_data_t max[] = {7};
  edm_data_t bin_spacing[] = {0.1};
  int periodic[] = {0};
  edm_data_t sigma[] = {0.1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  min[0] = 0;
  max[0] = 10;
  periodic[0] = 1;
  g.set_boundary(min, max, periodic);//test if we can set bounds first, then copy

  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  edm_data_t point[] = {0.01};
  edm_data_t* d_point;
  gpuErrchk(cudaMalloc((void**)&d_point, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(edm_data_t), cudaMemcpyHostToDevice));

//THE MOMENT WE'VE ALL BEEN WAITING FOR!!
  
//  g.add_value(point,1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_point, 1.0, d_g);

  edm_data_t* d_target;
  gpuErrchk(cudaMalloc((void**)&d_target, sizeof(edm_data_t)));//for checking the value
  edm_data_t der[1];
  point[0] = 0;
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));//for filling with deriv
  get_value_deriv_kernel<1><<<1,1>>>(d_point, d_der, d_target, &(d_g->grid_));//gross

  gpuErrchk(cudaMemcpy(&der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  

//  g.get_value_deriv(point, der);
  BOOST_REQUIRE(fabs(der[0]) > 0.1);

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_point));
  gpuErrchk(cudaFree(d_target));
  gpuErrchk(cudaFree(d_der));
}//gpu_boundary_remap_wrap_3

BOOST_AUTO_TEST_CASE( gpu_boundary_remap_nowrap_1) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  edm_data_t min[] = {-2};
  edm_data_t max[] = {7};
  edm_data_t bin_spacing[] = {0.1};
  int periodic[] = {0};
  edm_data_t sigma[] = {0.1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  min[0] = 0;
  max[0] = 10;
  periodic[0] = 0;
  g.set_boundary(min, max, periodic);
  
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));


  edm_data_t point[] = {-0.01};
  edm_data_t* d_point;
  gpuErrchk(cudaMalloc((void**)&d_point, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(edm_data_t), cudaMemcpyHostToDevice));
//  g.add_value(point,1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_point, 1.0, d_g);

//  edm_data_t der[1];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  point[0] = 0;
  gpuErrchk(cudaMemcpy(d_point, point, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  edm_data_t* d_target;
  gpuErrchk(cudaMalloc((void**)&d_target, sizeof(edm_data_t)));
//  g.get_value_deriv(point, der);
  get_value_deriv_kernel<1><<<1,1>>>(d_point, d_der, d_target, &(d_g->grid_));//gross
  gpuErrchk(cudaMemcpy(point, d_point, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(fabs(point[0]) < EPSILON);

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_point));
  gpuErrchk(cudaFree(d_target));
  gpuErrchk(cudaFree(d_der));

}//gpu_boundary_remap_nowrap_1

BOOST_AUTO_TEST_CASE( gpu_interp_3d_mixed ) {
  edm_data_t min[] = {-M_PI, -M_PI, 0};
  edm_data_t max[] = {M_PI, M_PI, 10};
  edm_data_t bin_spacing[] = {M_PI / 100, M_PI / 100, 1};
  int periodic[] = {1, 1, 0};
  DimmedGridGPU<3> g (min, max, bin_spacing, periodic, 1, 0);
  size_t index = 0;
  edm_data_t x,y,z;
  
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


  edm_data_t array[] = {-10.75 * M_PI / 2, 8.43 * M_PI / 2, 3.5};
  edm_data_t* d_array;
  gpuErrchk(cudaMalloc((void**)&d_array, 3*sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_array, array, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  edm_data_t der[3];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, 3*sizeof(edm_data_t)));
  edm_data_t* d_fhat;
  gpuErrchk(cudaMalloc((void**)&d_fhat, sizeof(edm_data_t)));
  get_value_deriv_kernel<3><<<1,1>>>(d_array, d_der, d_fhat, d_g);
  edm_data_t fhat[1];
  gpuErrchk(cudaMemcpy(fhat, d_fhat, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(der, d_der, 3*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
//  edm_data_t fhat = g.get_value_deriv(array,der);
  edm_data_t f = cos(array[0]) * sin(array[1]) * array[2];
  edm_data_t true_der[] = {-sin(array[0]) * sin(array[1]) * array[2],
		       cos(array[0]) * cos(array[1]) * array[2],
		       cos(array[0]) * sin(array[1])};
  
  BOOST_REQUIRE(pow(f- fhat[0], 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - true_der[0], 2) < 0.1);
  BOOST_REQUIRE(pow(der[1] - true_der[1], 2) < 0.1);
  BOOST_REQUIRE(pow(der[2] - true_der[2], 2) < 0.1);


  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_array));
  gpuErrchk(cudaFree(d_fhat));
  gpuErrchk(cudaFree(d_der));

}//gpu_interp_3d_mixed

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_add_check ) {
  edm_data_t min[] = {-10};
  edm_data_t max[] = {10};
  edm_data_t sigma[] = {1};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 0, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
  //add 1 gaussian
  edm_data_t x[] = {0};
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
//  g.add_value(x, 1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.0, d_g);

  //now check a few points
  edm_data_t* d_target;
  gpuErrchk(cudaMalloc((void**)&d_target, sizeof(edm_data_t)));
  get_value_kernel<1><<<1,1>>>(d_x, d_target, &(d_g->grid_));
  edm_data_t target[1];
  gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(pow(target[0] - 1 / sqrt(2 * M_PI), 2) < EPSILON);

  int i;
  edm_data_t der[1];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  for( i = -6; i < 7; i++) {
    x[0] = i;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
//    value = g.get_value_deriv(x, der);
    get_value_deriv_kernel<1><<<1, 1>>>(d_x, d_der, d_target, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(target, d_target, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    BOOST_REQUIRE(pow(target[0] - exp(-x[0]*x[0]/2.) / sqrt(2*M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-x[0] *exp(-x[0]*x[0]/2.)) / sqrt(2*M_PI), 2) < 0.01);
  }

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_target));
  gpuErrchk(cudaFree(d_der));

}//gpu_gauss_grid_add_check

BOOST_AUTO_TEST_CASE( gpu_gauss_pbc_check ) {
  edm_data_t min[] = {2};
  edm_data_t max[] = {10};
  edm_data_t sigma[] = {1};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 0, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add 1 gaussian
  edm_data_t x[] = {2};
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
//  g.add_value(x, 1);
  add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.0, d_g);
  
  int i;
  edm_data_t der[1];
  edm_data_t value[1];
  edm_data_t* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(edm_data_t)));
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  edm_data_t dx;//delta-x, not device_x
  for( i = -6; i < 7; i++) {
    x[0] = i;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    dx = x[0] - 2;
    dx -= round(dx / (min[0] - max[0])) * (min[0] - max[0]);
//    value = g.get_value_deriv(x, der);
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_value, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(value, d_value, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

    std::cout << "x = " << x[0]
	      << " dx = " << dx 
	      << "(" 
	      << " value = " << value[0]
	      << " (" << exp(-dx*dx/2.) / sqrt(2 * M_PI) << ")" 
	      << std::endl;

    BOOST_REQUIRE(pow(value[0] - exp(-dx*dx/2.) / sqrt(2 * M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-dx *exp(-dx*dx/2.)) / sqrt(2 * M_PI), 2) < 0.01);
  }

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_value));
  gpuErrchk(cudaFree(d_der));
 
}//gpu_gauss_pbc_check

BOOST_AUTO_TEST_CASE( gpu_gauss_subdivided_pbc_check ) {
  edm_data_t min[] = {2};
  edm_data_t max[] = {4};
  edm_data_t sigma[] = {1};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {0};
  edm_data_t gauss_loc[] = {11};
  edm_data_t* d_gauss_loc;
  gpuErrchk(cudaMalloc((void**)&d_gauss_loc, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_gauss_loc, gauss_loc, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  edm_data_t x[1];
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
  edm_data_t der[1];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  edm_data_t value[1];
  edm_data_t* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(edm_data_t)));
  edm_data_t dx;
  edm_data_t* d_x;//device copy of x
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(edm_data_t)));
  for( i = 2; i < 4; i++) {
    x[0] = i;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    dx = x[0] - gauss_loc[0];
    dx  -= round(dx / (min[0] - max[0])) * (min[0] - max[0]);
//    value = g.get_value_deriv(x, der);
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_value, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(value, d_value, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

    BOOST_REQUIRE(pow(value[0] - exp(-dx*dx/2.) / sqrt(2 * M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-dx *exp(-dx*dx/2.)) / sqrt(2 * M_PI), 2) < 0.01);
  }

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_value));
  gpuErrchk(cudaFree(d_der));
  gpuErrchk(cudaFree(d_gauss_loc));

}//gpu_gauss_subdivided_pbc_check

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_integral_test ) {
  edm_data_t min[] = {-100};
  edm_data_t max[] = {100};
  edm_data_t sigma[] = {1.2};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
  
  //add N gaussian
  int N = 20;
  int i;
  edm_data_t x[1];
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(edm_data_t)));
  edm_data_t offsets = 1. / N;
  edm_data_t g_integral[g.minisize_total_];
  edm_data_t* d_g_integral;
  gpuErrchk(cudaMalloc((void**)&d_g_integral, g.minisize_total_*sizeof(edm_data_t)));

  edm_data_t g_integral_total = 0;
  //generate a random number but use sequential grid point offsets
  int j;
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    add_value_integral_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.5, d_g_integral, d_g);
    gpuErrchk(cudaMemcpy(g_integral, d_g_integral, g.minisize_total_*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    for(j = 0; j < g.minisize_total_; j++){
      g_integral_total += g_integral[j];
    }
  }
  
  //now we integrate the grid
  edm_data_t area = 0;
  edm_data_t value[1] = {0};
  edm_data_t dx = 0.1;
  edm_data_t* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(edm_data_t)));
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(value, d_value, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    area += value[0] * dx;
  }

  //Make sure the integrated area is correct
  //unnormalized, so a little height scaling is necessary
  //  std::cout << area << " " << N * 1.5 << std::endl;
  BOOST_REQUIRE(pow(area - N * 1.5, 2) < 1);

  //now make sure that add_value returned the correct answers as well
  BOOST_REQUIRE(pow(area - g_integral_total, 2) < 0.1);

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_value));
  gpuErrchk(cudaFree(d_g_integral));
}//gpu_gauss_grid_integral_test

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_derivative_test ) {
  edm_data_t min[] = {-100};
  edm_data_t max[] = {100};
  edm_data_t sigma[] = {1.2};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add N gaussian
  int N = 20;
  int i;
  edm_data_t x[1];
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(edm_data_t)));
  
  edm_data_t offsets = 1. / N;
  
  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.5, d_g);
  }

  //now we calculate finite differences on the grid
  edm_data_t vlast, vlastlast, v, approx_der;
  edm_data_t* d_v;
  gpuErrchk(cudaMalloc((void**)&d_v, sizeof(edm_data_t)));

  edm_data_t der[1];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  edm_data_t der_last;
  edm_data_t dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_v, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(&v, d_v, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    if(i > 1) {
      approx_der = (v - vlastlast) / (2*dx);
      BOOST_REQUIRE(pow(approx_der - der_last, 2) < 0.01);
    }
    vlastlast = vlast;
    vlast = v;

    der_last = der[0];
  }

  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_v));
  gpuErrchk(cudaFree(d_der));

}//gpu_gauss_grid_derivative_test

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_derivative_test_mcgdp_1 ) {
  edm_data_t min[] = {-100};
  edm_data_t max[] = {100};
  edm_data_t sigma[] = {1.2};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  //add N gaussian
  int N = 20;
  int i;
  edm_data_t x[1];
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(edm_data_t)));
  edm_data_t offsets = 1. / N;

  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.5, d_g);
  }

  //now we calculate finite differences on the grid
  edm_data_t vlast, vlastlast, v, approx_der;
  edm_data_t* d_v;
  gpuErrchk(cudaMalloc((void**)&d_v, sizeof(edm_data_t)));


  edm_data_t der[1];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));
  edm_data_t der_last;
  edm_data_t dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_v, &(d_g->grid_));
    gpuErrchk(cudaMemcpy(&v, d_v, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

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


  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_v));
  gpuErrchk(cudaFree(d_der));

}//gpu_gauss_grid_derivative_test_mcgdp_1

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_interp_test_mcgdp_1D ) {
  edm_data_t min[] = {-100};
  edm_data_t max[] = {100};
  edm_data_t sigma[] = {10.0};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  periodic[0]  = 0;
  min[0] = -50;
  max[0] = 50;
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
  edm_data_t* d_min;
  edm_data_t* d_max;
  int* d_periodic;
  gpuErrchk(cudaMalloc((void**)&d_min, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_min, min, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_max, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_max, max, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_periodic, sizeof(int)));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, sizeof(int), cudaMemcpyHostToDevice));
  set_boundary_kernel<1><<<1,1>>>(d_min, d_max, d_periodic, d_g);



  //add N gaussian
  int N = 20;
  int i;
  edm_data_t x[1];
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, sizeof(edm_data_t)));
  edm_data_t der[1];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, sizeof(edm_data_t)));

  //generate a random number
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100;
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
    add_value_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.0, d_g);
  }

  //Check if the boundaries were duplicated
  edm_data_t value[1];
  edm_data_t value2[1];
  edm_data_t* d_value;
  gpuErrchk(cudaMalloc((void**)&d_value, sizeof(edm_data_t)));
  x[0] = 50.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value, d_value, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  x[0] = 49.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value2, d_value, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(value[0] - value2[0] ,2) < 2*(EPSILON));//This one cuts it too close...?

  x[0] = 150.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value, d_value, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  x[0] = 151.0;
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<1><<<1,1>>>(d_x, d_value, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(value2, d_value, sizeof(edm_data_t), cudaMemcpyDeviceToHost));

  BOOST_REQUIRE(pow(value[0] - value2[0] ,2) < EPSILON);

  x[0] = 50.0;

  //boundaries should be 0, even with interpolation
//  g.get_value_deriv(x,der);
  edm_data_t* d_dummy;
  gpuErrchk(cudaMalloc((void**)&d_dummy, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);

  //check other side
  x[0] = -50.1;
  x[0] = -50.0;
  gpuErrchk(cudaMalloc((void**)&d_dummy, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_deriv_kernel<1><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(der, d_der, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);


  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_min));
  gpuErrchk(cudaFree(d_max));
  gpuErrchk(cudaFree(d_periodic));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_value));
  gpuErrchk(cudaFree(d_dummy));
  gpuErrchk(cudaFree(d_der));

}//gpu_gauss_grid_interp_test_mcgdp_1D

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_interp_test_mcgdp_3D ) {
  edm_data_t min[] = {-10, -10, -10};
  edm_data_t* d_min;
  gpuErrchk(cudaMalloc((void**)&d_min, 3*sizeof(edm_data_t)));
  edm_data_t max[] = {10, 10, 10};
  edm_data_t* d_max;
  gpuErrchk(cudaMalloc((void**)&d_max, 3*sizeof(edm_data_t)));
  edm_data_t sigma[] = {3.0, 3.0, 3.0};
  edm_data_t bin_spacing[] = {0.9, 1.1, 1.4};
  int periodic[] = {1, 1, 1};
  int* d_periodic;
  gpuErrchk(cudaMalloc((void**)&d_periodic, 3*sizeof(int)));
  DimmedGaussGridGPU<3> g (min, max, bin_spacing, periodic, 1, sigma);
  periodic[0]  = periodic[1] = periodic[2] = 0;
  min[0]  = min[1] = min[2] = -5;
  max[0]  = max[1] = max[2] = 5;

  gpuErrchk(cudaMemcpy(d_min, min, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_max, max, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, 3*sizeof(int), cudaMemcpyHostToDevice));
  
//  g.set_boundary(min, max, periodic);


  DimmedGaussGridGPU<3>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<3>)));

  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<3>), cudaMemcpyHostToDevice));

  set_boundary_kernel<3><<<1,1>>>(d_min, d_max, d_periodic, d_g);

  //add N gaussian
  int N = 20;
  int i;
  edm_data_t x[3];
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc((void**)&d_x, 3*sizeof(edm_data_t)));
  edm_data_t der[3];
  edm_data_t* d_der;
  gpuErrchk(cudaMalloc((void**)&d_der, 3*sizeof(edm_data_t)));
  edm_data_t v;
  edm_data_t* d_v;
  gpuErrchk(cudaMalloc((void**)&d_v, 3*sizeof(edm_data_t)));

  //generate a random number
  for(i = 0; i < N; i++) {
    x[0] = rand() % 20 - 10;
    x[1] = rand() % 20 - 10;
    x[2] = rand() % 20 - 10;
    gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
    add_value_kernel<3><<<1, g.minisize_total_>>>(d_x, 5.0, d_g);
  }

  //Check if the boundaries were duplicated
  edm_data_t v2;
  x[0] = x[2] = 50.1;
  x[1] = 5.0;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v2, d_v, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  x[0] = x[1] = 50.0;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v, d_v, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  
  BOOST_REQUIRE(pow(v - v2, 2) < EPSILON);

  //boundaries should be 0, even with interpolation
  edm_data_t* d_dummy;
  gpuErrchk(cudaMalloc((void**)&d_dummy, sizeof(edm_data_t)));
//  g.get_value_deriv(x,der);
  get_value_deriv_kernel<3><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(der, d_der, 3*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < 0.001);

  //check another location
  x[0] = -5.1;
  x[2] = 5.1;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v, d_v, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  x[0] = x[2] = -5.0;
  gpuErrchk(cudaMemcpy(d_x, x, 3*sizeof(edm_data_t), cudaMemcpyHostToDevice));
  get_value_kernel<3><<<1,1>>>(d_x, d_v, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(&v2, d_v, sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(pow(v - v2, 2) < 0.001);
  get_value_deriv_kernel<3><<<1,1>>>(d_x, d_der, d_dummy, &(d_g->grid_));
  gpuErrchk(cudaMemcpy(der, d_der, 3*sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);


  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_min));
  gpuErrchk(cudaFree(d_max));
  gpuErrchk(cudaFree(d_periodic));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_v));
  gpuErrchk(cudaFree(d_dummy));
  gpuErrchk(cudaFree(d_der));

}//gpu_gauss_grid_interp_test_mcgdp_3D

BOOST_AUTO_TEST_CASE( gpu_gauss_grid_integral_regression_1 ) {
  edm_data_t min[] = {0};
  edm_data_t max[] = {10};
  edm_data_t bin_spacing[] = {0.009765625};
  edm_data_t sigma[] = {0.1};
  int periodic[] = {1};
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 1, sigma);
  periodic[0] = 1;
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));

  edm_data_t* d_min;
  gpuErrchk(cudaMalloc((void**)&d_min, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_min, min, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  edm_data_t* d_max;
  gpuErrchk(cudaMalloc((void**)&d_max, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_max, max, sizeof(edm_data_t), cudaMemcpyHostToDevice));
  int* d_periodic;
  gpuErrchk(cudaMalloc((void**)&d_periodic, sizeof(int)));
  gpuErrchk(cudaMemcpy(d_periodic, periodic, sizeof(int), cudaMemcpyHostToDevice));

  set_boundary_kernel<1><<<1,1>>>(d_min, d_max, d_periodic, d_g);
  
  //add gaussian that was failing
  edm_data_t x[1] = {-3.91944};
  edm_data_t h = 1.0;
  edm_data_t bias_added[g.minisize_total_]; //= g->add_value(x, h);
  edm_data_t* d_bias_added;
  gpuErrchk(cudaMalloc((void**)&d_bias_added, g.minisize_total_ * sizeof(edm_data_t)));
  edm_data_t* d_x;
  gpuErrchk(cudaMalloc(&d_x, sizeof(edm_data_t)));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(edm_data_t), cudaMemcpyHostToDevice));//here w/ limit=4096


  add_value_integral_kernel<1><<<1, g.minisize_total_>>>(d_x, 1.0, d_bias_added, d_g);
 gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(bias_added, d_bias_added, g.minisize_total_ * sizeof(edm_data_t), cudaMemcpyDeviceToHost));
  edm_data_t tot_bias_added = 0;
  for (int i = 0; i < g.minisize_total_; i++){
    tot_bias_added += bias_added[i];
  }
  
  //unnormalized, so a little height scaling is necessary
  //std::cout << bias_added /  (sqrt(2 * M_PI) * sigma[0]) << " " << h << std::endl;
  BOOST_REQUIRE(pow(tot_bias_added - h, 2) < 0.1);


  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_min));
  gpuErrchk(cudaFree(d_max));
  gpuErrchk(cudaFree(d_periodic));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_bias_added));

}//gpu_gauss_grid_integral_regression_1

BOOST_AUTO_TEST_CASE( edm_bias_reader_gpu ) {
  EDMBiasGPU bias(EDM_SRC + "/read_test.edm");
  BOOST_REQUIRE_EQUAL(bias.dim_, 2);
  BOOST_REQUIRE_EQUAL(bias.b_tempering_, 0);
  BOOST_REQUIRE(pow(bias.bias_sigma_[0] - 2,2) < EPSILON);
  BOOST_REQUIRE(pow(bias.bias_dx_[1] - 1.0,2) < EPSILON);
  printf("made it this far.\n");
}//edm_gpu_bias_reader

//this struct is used in the following sanity test
struct EDMBiasGPUTest{
  EDMBiasGPUTest() : bias(EDM_SRC + "/sanity.edm"){
    //subdivide is only called once at setup, on CPU-side
    bias.setup(1, 1);
    edm_data_t low[] = {0};
    edm_data_t high[] = {10};
    int p[] = {1};
    edm_data_t skin[] = {0};
    bias.subdivide(low, high, low, high, p, skin);
  }
  EDMBiasGPU bias;
};

BOOST_FIXTURE_TEST_SUITE( edmbiasgpu_test, EDMBiasGPUTest )

BOOST_AUTO_TEST_CASE( edm_gpu_sanity) {
  edm_data_t** positions = (edm_data_t**) malloc(sizeof(edm_data_t*));
  positions[0] = (edm_data_t*) malloc(sizeof(edm_data_t));
  edm_data_t runiform[] = {1};

  positions[0][0] = 5.0;
  bias.add_hills(1, positions, runiform);
  
  bias.write_bias("BIAS");

  //test if the value at the point is correct
  BOOST_REQUIRE(pow(bias.bias_->get_value(positions[0]) - bias.hill_prefactor_ / sqrt(2 * M_PI) / bias.bias_sigma_[0], 2) < EPSILON);
  //check if the claimed amount of bias added is correct
  BOOST_REQUIRE(pow(bias.cum_bias_ - bias.hill_prefactor_, 2) < 0.001);
  
  //now  check that the forces point away from the hills
  edm_data_t der[0];
  positions[0][0] = 4.99; //to the left
  bias.bias_->get_value_deriv(positions[0], der);
  //the negative of the bias (the force) should point to the left
  BOOST_REQUIRE(-der[0] < 0);

  positions[0][0] = 5.01; // to the right of the bias
  bias.bias_->get_value_deriv(positions[0], der);
  //the negative of the bias (the force) should point to the right
  BOOST_REQUIRE(-der[0] > 0);

  

  free(positions[0]);
  free(positions);
}

BOOST_AUTO_TEST_SUITE_END()

//This test will simply run several thousand timesteps and time how long it takes.
BOOST_AUTO_TEST_CASE( edm_cpu_timer_1d ){
  
  edm_data_t min[] = {-10};
  edm_data_t max[] = {10};
  edm_data_t sigma[] = {1};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {1};
  edm_data_t x[1] = {0};
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


//This test will simply run several thousand timesteps and time how long it takes.
BOOST_AUTO_TEST_CASE( edm_gpu_timer_1d ){
  
  edm_data_t min[] = {-10};
  edm_data_t max[] = {10};
  edm_data_t sigma[] = {1};
  edm_data_t bin_spacing[] = {1};
  int periodic[] = {1};
  unsigned int n_hills = 4096;
  DimmedGaussGridGPU<1> g (min, max, bin_spacing, periodic, 0, sigma);
  DimmedGaussGridGPU<1>* d_g;
  gpuErrchk(cudaMalloc((void**)&d_g, sizeof(DimmedGaussGridGPU<1>)));
  gpuErrchk(cudaMemcpy(d_g, &g, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
  edm_data_t* d_coordinates;
  gpuErrchk(cudaMalloc((void**)&d_coordinates, n_hills * sizeof(edm_data_t)));
  edm_data_t coordinates[n_hills];
  //now just call a kernel to add a bunch of gaussians, and time it
  boost::timer::auto_cpu_timer t;
  for( unsigned int i = 0; i < n_hills; i++){
    int rand_num = rand() % 20 - 10;
    coordinates[i] = rand_num;
  }
  gpuErrchk(cudaMemcpy(d_coordinates, coordinates, n_hills * sizeof(edm_data_t), cudaMemcpyHostToDevice));
  //add lots of hills in parallel!
  add_value_kernel<1><<<n_hills/8, 8 * g.minisize_total_>>>(d_coordinates, 1.0, d_g);
  gpuErrchk(cudaDeviceSynchronize());
  t.stop();
  sec seconds = chrono::nanoseconds(t.elapsed().user);
  
  BOOST_REQUIRE(seconds.count() < TIMING_BOUND_edm_cpu_timer_1d);
}

//BOOST_AUTO_TEST_SUITE_END()

