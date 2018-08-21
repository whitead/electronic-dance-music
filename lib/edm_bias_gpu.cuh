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
  __host__ inline int nextHighestPowerOf2(int input){
    int val = 1;
    while(val < input){
      val *= 2;
    }
    return(val);
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
    using EDMBias::write_bias;
    using EDMBias::bias_;//need same pointer for the write calls to work?
    using EDMBias::bias_dx_;
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
