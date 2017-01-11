#include "grid_gpu.cuh"
#include "edm_bias_gpu.cuh"
#include "gaussian_grid_gpu.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
//These must be declared here.
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE EDM_GPU

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

  for(int i = 0; i < 10; i++)
    g.grid_[i] = i;

  double x[] = {3.5};
  //check reading off of GPU
  BOOST_REQUIRE(g.in_grid(x));
  size_t index[1];
  g.get_index(x, index);
  BOOST_REQUIRE(index[0] - 3 < 0.000001);
  BOOST_REQUIRE(pow(g.get_value(x) -3, 2) < 0.000001);

  
  
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

