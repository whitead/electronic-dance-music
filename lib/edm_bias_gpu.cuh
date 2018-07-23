#ifndef EDM_BIAS_CH_
#define EDM_BIAS_CH_

#include <cuda_runtime.h>
#include <cuda.h>
#include "edm.h"
#include "edm_bias.h"
#include "gaussian_grid_gpu.cuh"

#define GPU_BIAS_BUFFER_SIZE 2048
#define GPU_BIAS_BUFFER_DBLS (2048 * 8)


namespace EDM{
  __host__ __device__ int nextHighestPowerOf2(int input){
    int val = 1;
    while(val < input){
      val *= 2;
    }
    return(val);
  }

//  template <unsigned int blockSize>
  __device__ void warpAddReduce(volatile edm_data_t *sdata, unsigned int tid, unsigned int blockSize){
    //This and addReduce must ALWAYS be called with a power-of-two size, even if the data
    //is smaller than that
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        __syncthreads();
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        __syncthreads();
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        __syncthreads();
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        __syncthreads();
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        __syncthreads();
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
        __syncthreads();
  }

//  template <unsigned int blockSize>
  __global__ void addReduce(edm_data_t *g_idata, edm_data_t *g_odata, unsigned int n, unsigned int blockSize){
    //This and warpAddReduce must ALWAYS be called with a power-of-two size, even if the data
    //is smaller than that
    extern __shared__ edm_data_t sdata[];//don't forget the size of sdata in the kernel call!
    unsigned int tid= threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[i] = 0;
    //printf("sdata[%i] is %f\n", i, sdata[i]);//all zeros: good
    __syncthreads();
    while (i < n){
      sdata[tid] += i+blockSize < n ? g_idata[i] + g_idata[i+blockSize] : g_idata[i];
      i += gridSize;
    }
    __syncthreads();

    if(blockSize >= 512){if (tid < 256){ sdata[tid] += sdata[tid + 256];} __syncthreads();}
    if(blockSize >= 256){if (tid < 128){ sdata[tid] += sdata[tid + 128];} __syncthreads();}
    if(blockSize >= 128){if (tid < 64){ sdata[tid] += sdata[tid + 64];} __syncthreads();}
    if(tid < 32) warpAddReduce(sdata,tid, blockSize);//<blockSize>(sdata, tid);
    __syncthreads();
    __syncthreads();
    if(tid == 0) {g_odata[0] = sdata[0];}
    __syncthreads();
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
    void post_add_hill();

    int read_input(const std::string& input_filename);
    edm_data_t flush_buffers(int synched);
    edm_data_t do_add_hills(const edm_data_t* buffer, const size_t hill_number, char hill_type);
    edm_data_t* send_buffer_;
    int minisize;//tracks the minisize of the bias
    edm_data_t* d_bias_added_;//track bias added on GPU.

    void output_hill(const edm_data_t* position, edm_data_t height, edm_data_t bias_added, char type);

    
  private:
    //histogram output
    std::string hist_output_;
    void launch_add_value_integral_kernel(int dim, const edm_data_t* buffer, edm_data_t* target, Grid* grid, dim3 grid_dims);

  };

}
#endif // EDM_BIAS_CH_
