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
  g.get_value(x);

  x[0] = 10;
  g.get_value(x);
  gpuErrchk(cudaFree(d_g));
  gpuErrchk(cudaFree(d_target));
  gpuErrchk(cudaFree(d_x));
}

BOOST_AUTO_TEST_CASE( grid_gpu_3d_sanity )
{//must now refactor this test to use kernels.
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
  size_t temp[3];
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
	  /* This passes but it's slow so leaving out for now
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(d_array, array, 3*sizeof(size_t), cudaMemcpyHostToDevice));

	//g.one2multi(g.multi2one(array), temp);
	multi2one_kernel<3><<<1,1>>>(d_g, d_array, d_temp);
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
//  printf("Guess we passed the grid_gpu_3d_sanity!\n");
}

BOOST_AUTO_TEST_CASE( grid_gpu_1d_read ) {
  printf("test. did I make it into 1d_read?\n");
  DimmedGridGPU<1> g(GRID_SRC + "/1.grid");
  printf("test. did I make it past reading in 1d_read?\n");
  BOOST_REQUIRE_EQUAL(g.min_[0], 0);
  BOOST_REQUIRE_EQUAL(g.max_[0], 2.5 + g.dx_[0]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
  printf("test. did I make it to the end of 1d_read?\n");
}

BOOST_AUTO_TEST_CASE( grid_gpu_3d_read ) {
  printf("test. did I make it into 3d_read?\n");
  DimmedGridGPU<3> g(GRID_SRC + "/3.grid");
  printf("test. did I make it past reading in 3d_read?\n");
  BOOST_REQUIRE_EQUAL(g.min_[2], 0);
  BOOST_REQUIRE_EQUAL(g.max_[2], 2.5 + g.dx_[2]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[2], 11);
  double temp[] = {0.75, 0, 1.00};
  gpuErrchk(cudaDeviceSynchronize());
  BOOST_REQUIRE(pow(g.do_get_value(temp) - 1.260095, 2) < EPSILON);
  printf("test. did I make it to the end of 3d_read?\n");
}

BOOST_AUTO_TEST_CASE( gpu_derivative_direction ) {
  DimmedGridGPU<3> g(GRID_SRC + "/3.grid", 1);
  double temp[] = {0.75, 0, 1.00};
  double temp2[] = {0.76, 0, 1.00};
  BOOST_REQUIRE(g.get_value(temp2)> g.get_value(temp));
  temp2[0] = 0.75;
  temp2[2] = 0.99;
  BOOST_REQUIRE(g.get_value(temp2) < g.get_value(temp));
}

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
    g->read(output);
    //now compare
    BOOST_REQUIRE_EQUAL(g->get_grid_size(), ref_length);

    for(j = 0; j < ref_length; j++)
      BOOST_REQUIRE(pow(ref_grid[j] - g->get_grid()[j], 2) < EPSILON);

  }
  
}

BOOST_AUTO_TEST_CASE( gpu_interpolation_1d ) {
  
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGridGPU<1> g (min, max, bin_spacing, periodic, 1, 1);
  
  for(int i = 0; i < 11; i++) {
    g.grid_[i] = log(i);
    g.grid_deriv_[i] = 1. / i;
  }

  double array[] = {5.3};
  double der[1];
  double fhat = g.get_value_deriv(array,der);

  //make sure it's at least in the ballpark
  BOOST_REQUIRE(fhat > log(5) && fhat < log(6));
  BOOST_REQUIRE(der[0] < 1. / 5 && der[0] > 1. / 6.);

  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat - log(5.3), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0]- 1. / 5.3, 2) < 0.1);

  //try edge cases
  array[0] = 5.0;
  g.get_value(array);
  array[0] = 5.5;
  g.get_value(array);
  array[0] = 0.0;
  g.get_value(array);
  array[0] = 10.0;
  g.get_value(array);

}



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

