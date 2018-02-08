#ifndef EDM_BIAS_CH_
#define EDM_BIAS_CH_

#include <cuda_runtime.h>
#include <cuda.h>
#include "edm.h"
#include "edm_bias.h"
#include "gaussian_grid_gpu.cuh"

namespace EDM{ 

//  template <unsigned int blockSize>
  __device__ void warpAddReduce(volatile edm_data_t *sdata, unsigned int tid, unsigned int blockSize){
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
  }

//  template <unsigned int blockSize>
  __global__ void addReduce(edm_data_t *g_idata, edm_data_t *g_odata, unsigned int n, unsigned int blockSize){
    printf("Hello from thread %d\n", threadIdx.x);
    extern __shared__ edm_data_t sdata[];
    unsigned int tid= threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n){sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize;}
    __syncthreads();

    if(blockSize >= 512){if (tid < 256){ sdata[tid] += sdata[tid + 256];} __syncthreads();}
    if(blockSize >= 512){if (tid < 128){ sdata[tid] += sdata[tid + 128];} __syncthreads();}
    if(blockSize >= 512){if (tid < 64){ sdata[tid] += sdata[tid + 64];} __syncthreads();}

    if(tid < 32) warpAddReduce(sdata,tid, blockSize);//<blockSize>(sdata, tid);
    if(tid == 0) g_odata[blockIdx.x] = sdata[0];
  }

  template <unsigned int blockSize>
  __global__ void gpu_add_matrices(edm_data_t *MatA, edm_data_t *MatB, edm_data_t *MatC, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;
    if(ix < nx && iy < ny){
      MatC[idx] = MatA[idx] + MatB[idx];
    }
  }



  class EDMBiasGPU : public EDMBias {
    /** The EDM bias class. The main biasing class
     *
     *
     */
  public:

    EDMBiasGPU(const std::string& input_filename);
    ~EDMBiasGPU();
    void subdivide(const edm_data_t sublo[3], const edm_data_t subhi[3], 
		   const edm_data_t boxlo[3], const edm_data_t boxhi[3],
		   const int b_periodic[3], const edm_data_t skin[3]);

    void add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform);
    void add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform, int apply_mask);
    void queue_add_hill(const edm_data_t* position, edm_data_t this_h);
    using EDMBias::pre_add_hill;

    int read_input(const std::string& input_filename);

    edm_data_t* send_buffer_;

    
  private:
    //histogram output
    std::string hist_output_;


  };

}
#endif // EDM_BIAS_CH_
