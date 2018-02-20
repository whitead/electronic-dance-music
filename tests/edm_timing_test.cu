#include "grid_gpu.cuh"
#include "edm_bias.h"
#include "edm.h"
#include "edm_bias_gpu.cu"
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

//this struct is used in the following sanity test
struct EDMBiasGPUTest{
  EDMBiasGPUTest() : bias_gpu(EDM_SRC + "/sanity.edm"){
    //subdivide is only called once at setup, on CPU-side
    bias_gpu.setup(1, 1);
    edm_data_t low[] = {0};
    edm_data_t high[] = {10};
    int p[] = {1};
    edm_data_t skin[] = {0};
    bias_gpu.subdivide(low, high, low, high, p, skin);
  }
  EDMBiasGPU bias_gpu;
};
struct EDMBiasTest{
 
  EDMBiasTest() : bias(EDM_SRC + "/sanity.edm") {
        
    bias.setup(1, 1);
    edm_data_t low[] = {0};
    edm_data_t high[] = {10};
    int p[] = {1};
    edm_data_t skin[] = {0};
    bias.subdivide(low, high, low, high, p, skin);
    
  }     
  
  EDMBias bias;
 
};

BOOST_FIXTURE_TEST_SUITE( edmbiasgpu_test, EDMBiasGPUTest )

BOOST_AUTO_TEST_CASE( edm_timing_gpu) {
  using namespace std;
  ofstream output;
  string filename = "timing_gpu_log.txt";
  output.open(filename.c_str(),std::ios::ate);
  int n_hills = 4096;
  printf("beginning coordinates allocation (GPU)...\n");
  edm_data_t** coordinates = (edm_data_t**) malloc(4096*1024*2*sizeof(edm_data_t*));
  for (int i = 0; i < n_hills*2*1024; i++){
    coordinates[i] = (edm_data_t*) malloc(sizeof(edm_data_t));    
  }
  printf("allocating runiform (GPU)\n");
  edm_data_t* runiform = (edm_data_t*)malloc(4096*2*1024*sizeof(edm_data_t));

  sec seconds_gpu = chrono::nanoseconds(0);
  while(n_hills <= (4096*1024)){
    cout << endl << "NHILLS: " << n_hills << endl;
    output << "NHILLS: " << n_hills << endl;
    boost::timer::auto_cpu_timer t_gpu("%t sec CPU, %w sec real");
    //set coordinates randomly, but always add.
    printf("Setting coordinate values...\n");
    for( unsigned int i = 0; i < 2*n_hills; i+=2){
      int rand_num = rand() % 20 - 10;
      coordinates[i][0] = rand_num;
      runiform[i] = 1.0;
    }
    bias_gpu.add_hills(n_hills, coordinates, runiform);//add hills on gpu

    seconds_gpu = chrono::nanoseconds(t_gpu.elapsed().user);

    cout << "GPU TIME: " << seconds_gpu << endl;
    output << "GPU TIME: " << seconds_gpu << endl;
    printf("Finished the iteration with %zd hills on GPU\n", n_hills);
    n_hills *= 2;//more hills next time
  }
  //now just call a kernel to add a bunch of gaussians, and time it
  printf("\nAll done on GPU! n_hills = %zd\n", n_hills);
  output.close();
  free(coordinates);
  free(runiform);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_FIXTURE_TEST_SUITE( edmbias_test, EDMBiasTest )
BOOST_AUTO_TEST_CASE( edm_timing_cpu) {
  using namespace std;
  ofstream output;
  string filename = "timing_cpu_log.txt";
  output.open(filename.c_str(), std::ios::ate);
  unsigned int n_hills = 4096;
  printf("beginning coordinates allocation (CPU)...\n");
  edm_data_t** coordinates = (edm_data_t**) malloc(4096*1024*2*sizeof(edm_data_t*));
  for (int i = 0; i < n_hills*2*1024; i++){
    coordinates[i] = (edm_data_t*) malloc(sizeof(edm_data_t));    
  }
  printf("allocating runiform (CPU)\n");
  edm_data_t* runiform = (edm_data_t*)malloc(4096*2*1024*sizeof(edm_data_t));

  sec seconds_cpu = chrono::nanoseconds(0);
  while(n_hills <= (4096*1024)){
    cout << endl << "NHILLS: " << n_hills << endl;
    output << "NHILLS: " << n_hills << endl;
    boost::timer::auto_cpu_timer t_cpu("%t sec CPU, %w sec real");
    //set coordinates randomly, but always add.
    for( unsigned int i = 0; i < 2*n_hills; i+=2){
      int rand_num = rand() % 10;
      coordinates[i][0] = rand_num;
      runiform[i] = 1.0;
    }

    bias.add_hills(n_hills, coordinates, runiform);

    seconds_cpu = chrono::nanoseconds(t_cpu.elapsed().user);

    cout << "CPU TIME: " << seconds_cpu << endl;
    output << "CPU TIME: " << seconds_cpu << endl;

    printf("Finished the iteration with %zd hills on CPU\n", n_hills);
    n_hills *= 2;//more hills next time
  }
  //now just call a kernel to add a bunch of gaussians, and time it
  printf("\nAll done on CPU! n_hills = %zd\n", n_hills);
  output.close();
  free(coordinates);
  free(runiform);
}

BOOST_AUTO_TEST_SUITE_END()






