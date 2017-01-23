#include "grid_gpu.cuh"
#include "edm_bias_gpu.cuh"
#include "gaussian_grid_gpu.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
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

typedef chrono::duration<double> sec; // seconds, stored with a double

//Many of these test are the same as the serial ones, just to make sure we preserve behavior
BOOST_AUTO_TEST_CASE( grid_gpu_1d_sanity ){
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGridGPU<1> g (min, max, bin_spacing, periodic, 0, 0);

  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 11);
  BOOST_REQUIRE_EQUAL(g.grid_size_, 11);

  size_t array[] = {5};
  size_t temp[1];
  g.one2multi(g.multi2one(array), temp);
  BOOST_REQUIRE_EQUAL(array[0], temp[0]);

  for(int i = 0; i < 11; i++)
    g.grid_[i] = i;
  double x[] = {3.5};
  //check reading off of GPU
  BOOST_REQUIRE(g.in_grid(x));
  size_t index[1];
  g.get_index(x, index);
  BOOST_REQUIRE(index[0] - 3 < 0.000001);

  BOOST_REQUIRE(pow(g.get_value(x) -3, 2) < 0.000001);

  //try to break it
  x[0] = 0;
  g.get_value(x);

  x[0] = 10;
  g.get_value(x);

}

BOOST_AUTO_TEST_CASE( grid_gpu_3d_sanity )
{
  double min[] = {-2, -5, -3};
  double max[] = {125, 63, 78};
  double bin_spacing[] = {1.27, 1.36, 0.643};
  int periodic[] = {0, 1, 1};
  DimmedGridGPU<3> g (min, max, bin_spacing, periodic, 0, 0);

  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
  BOOST_REQUIRE_EQUAL(g.grid_number_[1], 50);
  BOOST_REQUIRE_EQUAL(g.grid_number_[2], 126);

  size_t array[3];
  size_t temp[3];
  for(int i = 0; i < g.grid_number_[0]; i++) {
    array[0] = i;
    for(int j = 0; j < g.grid_number_[1]; j++) {
      array[1] = j;
      for(int k = 0; k < g.grid_number_[2]; k++) {
	array[2] = k;

	//check to make sure the index conversion is correct
	g.one2multi(g.multi2one(array), temp);
	BOOST_REQUIRE_EQUAL(array[0], temp[0]);
	BOOST_REQUIRE_EQUAL(array[1], temp[1]);
	BOOST_REQUIRE_EQUAL(array[2], temp[2]);

	g.grid_[g.multi2one(array)] = g.multi2one(array);
      }
    }
  }
  
  double point[3];
  for(int i = 0; i < g.grid_number_[0]; i++) {
    point[0] = i * g.dx_[0] + g.min_[0] + EPSILON;
    array[0] = i;
    for(int j = 0; j < g.grid_number_[1]; j++) {
      point[1] = j * g.dx_[1] + g.min_[1] + EPSILON;
      array[1] = j;
      for(int k = 0; k < g.grid_number_[2]; k++) {
	point[2] = k * g.dx_[2] + g.min_[2] + EPSILON;
	array[2] = k;

	BOOST_REQUIRE(pow(g.get_value(point) - g.grid_[g.multi2one(array)],2) < 0.0000001);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( grid_1d_read ) {
  DimmedGridGPU<1> g(GRID_SRC + "/1.grid");
  BOOST_REQUIRE_EQUAL(g.min_[0], 0);
  BOOST_REQUIRE_EQUAL(g.max_[0], 2.5 + g.dx_[0]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
}

BOOST_AUTO_TEST_CASE( grid_3d_read ) {
  DimmedGridGPU<3> g(GRID_SRC + "/3.grid");
  BOOST_REQUIRE_EQUAL(g.min_[2], 0);
  BOOST_REQUIRE_EQUAL(g.max_[2], 2.5 + g.dx_[2]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[2], 11);
  double temp[] = {0.75, 0, 1.00};
  BOOST_REQUIRE(pow(g.get_value(temp) - 1.260095, 2) < EPSILON);
}

BOOST_AUTO_TEST_CASE( derivative_direction ) {
  DimmedGridGPU<3> g(GRID_SRC + "/3.grid");
  g.b_interpolate_ = 1;

  double temp[] = {0.75, 0, 1.00};
  double temp2[] = {0.76, 0, 1.00};
  BOOST_REQUIRE(g.get_value(temp2)> g.get_value(temp));
  temp2[0] = 0.75;
  temp2[2] = 0.99;
  BOOST_REQUIRE(g.get_value(temp2) < g.get_value(temp));
  
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

