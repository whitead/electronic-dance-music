#include "edm_bias_gpu.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <iterator>
#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <map>

namespace EDM_Kernels{
  //the kernels exist for testing purposes.
  using namespace EDM;

  inline __device__ void warpAddReduce(volatile edm_data_t *sdata, unsigned int tid, unsigned int blockSize){
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

  inline __global__ void addReduce(edm_data_t *g_idata, edm_data_t *g_odata, unsigned int n, unsigned int blockSize){
    //This and warpAddReduce must ALWAYS be called with a power-of-two size, even if the data
    //is smaller than that
    extern __shared__ edm_data_t sdata[];//don't forget the size of sdata in the kernel call!
    unsigned int tid= threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[i] = 0;
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

  /*
   * Kernel wrapper for do_add_value() on the GPU. Takes in a point, the hill height to add, 
   * an instance of DimmedGaussGridGPU to do the adding, as well as a pointer to a edm_data_t array
   * of size equal to the gaussian support where the total height added (hill integral) 
   * will be stored.
   */

  template <int DIM>
  __global__ void add_value_integral_kernel(const edm_data_t* buffer, edm_data_t* target, DimmedGaussGridGPU<DIM>* g){
    target[threadIdx.x] = g->do_add_value(buffer);
    return;
  }
}


using namespace EDM_Kernels;

EDM::EDMBiasGPU::EDMBiasGPU(const std::string& input_filename) : EDMBias(input_filename),
                                                                 d_bias_added_(NULL){
  gpuErrchk(cudaMallocManaged(&send_buffer_, sizeof(edm_data_t) * GPU_BIAS_BUFFER_SIZE));
  for(int i = 0; i < GPU_BIAS_BUFFER_DBLS; i++){
    send_buffer_[i] = 0;//CUDA wants this..?
  }
  read_input(input_filename);
  gpuErrchk(cudaDeviceSynchronize());
}


EDM::EDMBiasGPU::~EDMBiasGPU() {
  gpuErrchk(cudaDeviceSynchronize());
  /* 
   * target_, bias_dx_, and cv_hist_ are all delete'd by the superclass, EDMBias
   */
  if(bias_dx_ != NULL){
    gpuErrchk(cudaFree(bias_dx_));
    gpuErrchk(cudaDeviceSynchronize());
    bias_dx_ = NULL;
  }
    
  if(bias_sigma_ != NULL){
    gpuErrchk(cudaFree(bias_sigma_));
    gpuErrchk(cudaDeviceSynchronize());
    bias_sigma_ = NULL;
  }
    
  if(min_ != NULL){
    gpuErrchk(cudaFree(min_));
    gpuErrchk(cudaDeviceSynchronize());
    min_ = NULL;
  }
    
  if(max_ != NULL){
    gpuErrchk(cudaFree(max_));
    gpuErrchk(cudaDeviceSynchronize());
    max_ = NULL;
  }
    
  if(b_periodic_boundary_ != NULL){
    gpuErrchk(cudaFree(b_periodic_boundary_));
    gpuErrchk(cudaDeviceSynchronize());
    b_periodic_boundary_ = NULL;
  }
  if(send_buffer_ != NULL){
    gpuErrchk(cudaFree(send_buffer_));
    gpuErrchk(cudaDeviceSynchronize());
    send_buffer_ = NULL;
  }

  if(d_bias_added_ != NULL){
    gpuErrchk(cudaFree(d_bias_added_));
    gpuErrchk(cudaDeviceSynchronize());
    d_bias_added_ = NULL;
  }
  gpuErrchk(cudaDeviceSynchronize());
}

void EDM::EDMBiasGPU::subdivide(const edm_data_t sublo[3], 
			     const edm_data_t subhi[3], 
			     const edm_data_t boxlo[3],
			     const edm_data_t boxhi[3],
			     const int b_periodic[3],
			     const edm_data_t skin[3]) {

  //has subdivide already been called?
  if(bias_ != NULL)
    return;

  //has setup been called?
  if(temperature_ < 0)
    edm_error("Must call setup before subdivide", "edm_bias.cpp:subdivide");
  
  int grid_period[] = {0, 0, 0};
  edm_data_t min[3];
  edm_data_t max[3];
  size_t i;
  int bounds_flag = 1;

  for(i = 0; i < dim_; i++) {
    b_periodic_boundary_[i] = 0;
    //check if the given boundary matches the system boundary, if so then use the system periodicity
    if(fabs(boxlo[i] - min_[i]) < 0.000001 && fabs(boxhi[i] - max_[i]) < 0.000001)
      b_periodic_boundary_[i] = b_periodic[i];
    
  }

  for(i = 0; i < dim_; i++) {

    min[i] = sublo[i];      
    max[i] = subhi[i];      

    //check if we encapsulate the entire bounds in any dimension
    if(fabs(sublo[i] - min_[i]) < 0.000001 && fabs(subhi[i] - max_[i]) < 0.000001) {
      grid_period[i] = b_periodic[i];
      bounds_flag = 0;      
    } else {
      min[i] -= skin[i];
      max[i] += skin[i];
    }
      
    //check if we'll always be out of bounds
    bounds_flag &= (min[i] >= max_[i] || max[i] <= min_[i]);    
    
  }

  bias_ = make_gauss_grid_gpu(dim_, min, max, bias_dx_, grid_period, INTERPOLATE, bias_sigma_);
  //create histogram with no interpolation/no derivatives for tracking CV
  cv_hist_ = make_grid_gpu(dim_, min, max, bias_sigma_, grid_period, 0, 0);
    
  bias_->set_boundary(min_, max_, b_periodic_boundary_);
  if(initial_bias_ != NULL)
    bias_->add(initial_bias_, 1.0, 0.0);

  size_t free[1];
  size_t total[1];
  minisize = bias_->get_minisize_total();
//  cudaMemGetInfo(free, total);
//  printf("Free device mem: %zd // Total device mem: %zd // Attempting to malloc: %zu\n", free, total, minisize * GPU_BIAS_BUFFER_SIZE * sizeof(edm_data_t));
  gpuErrchk(cudaMallocManaged((void**)&d_bias_added_, minisize * GPU_BIAS_BUFFER_SIZE * sizeof(edm_data_t)));//that's the biggest it will have to be
  gpuErrchk(cudaMemset(d_bias_added_, 0, minisize * GPU_BIAS_BUFFER_SIZE * sizeof(edm_data_t)));
//  for(int i = 0; i < minisize * GPU_BIAS_BUFFER_SIZE; i++){
//    d_bias_added_[i] = 0;
//  }

  

  
  if(bounds_flag) {
    //we do this after so that we have a grid to at least write out
    std::cout << "I am out of bounds!" << std::endl;
    b_outofbounds_ = 1;
    return;
  }

  //get volume
  //note that get_volume won't get the system volume, due the skin 
  //between regions. However, it is correct for getting average bias
  //because some hills will be counted twice and this increase in volume
  //compensates for that.
  edm_data_t other_vol = 0;
  edm_data_t vol = bias_->get_volume();
  total_volume_ = 0;

  other_vol = vol;
  total_volume_ += other_vol;
}

int EDM::EDMBiasGPU::read_input(const std::string& input_filename){ 
  /* Overrides the EDMBias read_input method so that we allocate managed memory 
   * where appropriate, etc.
   */
  //parse file into a map
  using namespace std;
  ifstream input(input_filename.c_str());
  if(!input.is_open()) {      
    cerr << "Cannot open input file " << input_filename << endl;
    return 0;
  }

  map<string, string> parsed_input;
 
  insert_iterator< map<string, string> > mpsi(parsed_input, parsed_input.begin());
 
  const istream_iterator<pair<string,string> > eos; 
  istream_iterator<pair<string,string> > its (input);
 
  copy(its, eos, mpsi);

  //now convert key value pairs
  if(!extract_int("tempering", parsed_input, 1, &b_tempering_)) {
    cerr << "Must specify if tempering is enabled, ex: tempering 1 or tempering 0" << endl;
    return 0;
  }

  if(b_tempering_) {
    if(!extract_edm_data_t("bias_factor", parsed_input, 1,&bias_factor_))
      return 0;
    extract_edm_data_t("global_tempering", parsed_input, 0,&global_tempering_);    
  }

  if(!extract_edm_data_t("hill_prefactor", parsed_input, 1, &hill_prefactor_))
    return 0;
  extract_edm_data_t("hill_density", parsed_input, 0, &hill_density_);
  int tmp;
  if(!extract_int("dimension", parsed_input, 1, &tmp))
    return 0;
  else
    dim_ = tmp;
  if(dim_ == 0 || dim_ > 3) {
    cerr << "Invalid dimesion " << dim_ << endl;
    return 0;
  }
    

  //parse arrays now
  if(bias_dx_ != NULL)
    free(bias_dx_);
  cudaMallocManaged(&bias_dx_, sizeof(edm_data_t) * dim_);
  cudaMallocManaged(&bias_sigma_ ,sizeof(edm_data_t) * dim_ );
  cudaMallocManaged(&min_ ,sizeof(edm_data_t) * dim_ );
  cudaMallocManaged(&max_ ,sizeof(edm_data_t) * dim_ );
  cudaMallocManaged(&b_periodic_boundary_ ,sizeof(int) * dim_ );
  if(!extract_edm_data_t_array("bias_spacing", parsed_input, 1, bias_dx_, dim_))
    return 0;
  if(!extract_edm_data_t_array("bias_sigma", parsed_input, 1, bias_sigma_, dim_))
    return 0;
  if(!extract_edm_data_t_array("box_low", parsed_input, 1, min_, dim_))
    return 0;
  if(!extract_edm_data_t_array("box_high", parsed_input, 1, max_, dim_))
    return 0;

  //get target
  if(parsed_input.find("target_filename") == parsed_input.end()) {
    b_targeting_ = 0;
    expected_target_ = 0;
  }
  else {
    b_targeting_ = 1;
    string tfilename = parsed_input.at("target_filename");
    string cleaned_filename = clean_string(tfilename, 0);
    target_ = read_grid_gpu(dim_, cleaned_filename, 0); //read grid, do not use interpolation
    expected_target_ = target_->expected_bias();//read_grid_gpu() returns a Grid*
    std::cout << "(edm_bias_gpu) Expected Target is " << expected_target_ << std::endl;
  }
  if(parsed_input.find("initial_bias_filename") == parsed_input.end()) {
    initial_bias_ = NULL;
  } else {
    string ibfilename = parsed_input.at("initial_bias_filename");
    string cleaned_filename = clean_string(ibfilename, 0);
    initial_bias_ = read_grid_gpu(dim_, cleaned_filename, 1); //read grid, do use interpolation
  }

  if(parsed_input.find("hills_filename") != parsed_input.end()) {
    string hfilename = parsed_input.at("hills_filename");
    string cleaned_filename = clean_string(hfilename, 1);
    if(!(hill_output_.is_open())){
      hill_output_.open(cleaned_filename.c_str());    
    }
  }
  else {
    string hfilename("HILLS");
    string cleaned_filename = clean_string(hfilename, 1);
    if(!(hill_output_.is_open())){
      hill_output_.open(cleaned_filename.c_str());
    }

  }
  if(parsed_input.find("histogram_filename") != parsed_input.end()) {
    string hist_filename = parsed_input.at("histogram_filename");
    hist_output_ = clean_string(hist_filename, 0);
  } else {
    string hist_filename("HIST");
      hist_output_ = clean_string(hist_filename, 0);
  }

  return 1;
}

void EDM::EDMBiasGPU::add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform) {
  add_hills(nlocal, positions, runiform, -1);
}

//TODO (maybe): parallelize this call too
void EDM::EDMBiasGPU::add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform, int apply_mask) {

  int i;
  pre_add_hill(nlocal);//this is unchanged b/c it's all cpu-side
  for(i = 0; i < nlocal; i++) {
    if(apply_mask < 0 || apply_mask & mask_[i])
      add_hill(&positions[i][0], runiform[i]);//this is unchanged b/c it's all cpu-side
  }
  post_add_hill();

}

void EDM::EDMBiasGPU::post_add_hill() {

  if(temp_hill_cum_ < 0) {
    //error must call pre_add_hill before post_add_hill
  }

  temp_hill_cum_ += flush_buffers(1); //flush and done


  update_height(temp_hill_cum_);

  temp_hill_cum_ = -1;
  temp_hill_prefactor_ = -1;
  steps_++;
}


void EDM::EDMBiasGPU::queue_add_hill(const edm_data_t* position, edm_data_t this_h){
  //use the same buffer system but there's only one cause it's just GPU
  size_t i;
  for(i = 0; i < dim_; i++)
    send_buffer_[buffer_i_ * (dim_+ 1) + i] = position[i];
  send_buffer_[buffer_i_ * (dim_ + 1) + i] = this_h;
  buffer_i_++;
  
  //do we need to flush?
  if(buffer_i_ >= GPU_BIAS_BUFFER_SIZE)
    temp_hill_cum_ += flush_buffers(0); //flush and we don't know if we're synched
  
}

edm_data_t EDM::EDMBiasGPU::flush_buffers(int synched) {
  edm_data_t bias_added = 0;
  //flush the buffer. only one with GPU version for now...
  //TODO: make this a kernel call!
  gpuErrchk(cudaDeviceSynchronize());
  bias_added += do_add_hills(send_buffer_, buffer_i_, ADD_HILL);

  //reset buffer count
  buffer_i_ = 0;

  return bias_added;
}

edm_data_t EDM::EDMBiasGPU::do_add_hills(const edm_data_t* buffer, const size_t hill_number, char hill_type){
  edm_data_t bias_added = 0;
  size_t i, j;
  dim3 grid_dims(minisize, hill_number);
  launch_add_value_integral_kernel(dim_, buffer, d_bias_added_, bias_, grid_dims);//this launches kernel.
  gpuErrchk(cudaDeviceSynchronize());
//  for(i = 0; i < hill_number; i++){
    // for(j = 0; j < minisize; j++){
    //   bias_added += d_bias_added_[i*minisize + j];//sum over each mini-grid
    // }
    //have to call each addreduce separately for this bookkeeping
  addReduce<<<1,  nextHighestPowerOf2(minisize * hill_number),  nextHighestPowerOf2(minisize * hill_number) * sizeof(edm_data_t)>>>(d_bias_added_, d_bias_added_, minisize * hill_number,  nextHighestPowerOf2(minisize * hill_number));
    gpuErrchk(cudaDeviceSynchronize());
    bias_added += d_bias_added_[0];//[i*minisize];
    //TODO: Fix this so we can output hills again
//    output_hill(&buffer[i * (dim_ + 1)], buffer[i * (dim_ + 1) + dim_], bias_added, hill_type);
//  }
  hills_added_ += hill_number;
  return bias_added;
}

void EDM::EDMBiasGPU::output_hill(const edm_data_t* position, edm_data_t height, edm_data_t bias_added, char type) {
  
  size_t i;
  
  hill_output_ << std::setprecision(8) << std::fixed;
  hill_output_ << steps_ << " ";
  hill_output_ << type << " ";
  hill_output_ << hills_added_ << " ";
  for(i = 0; i < dim_; i++)  {
    hill_output_ << position[i] << " ";
  }
  hill_output_ << height << " ";
  hill_output_ << bias_added << " ";
  hill_output_ << cum_bias_ / total_volume_ << std::endl;

  //histogram it
  if(type == NEIGH_HILL ||
     type == BUFF_HILL ||
     type == ADD_HILL) {
    //cv_hist_->add_value(position, 1);
  } else if(type == ADD_UNDO_HILL ||
	    type == BUFF_UNDO_HILL) {
    //undo histogram
    //cv_hist_->add_value(position, -1);//commented for timing test only
  }
   
}

/*have to use another switch-based function to dynamically launch 
 *a kernel without prior knowledge of what dimension we're using.
 */
void EDM::EDMBiasGPU::launch_add_value_integral_kernel(int dim, const edm_data_t* buffer, edm_data_t* target, Grid* bias, dim3 grid_dims){
  switch(dim){
  case 1:
    DimmedGaussGridGPU<1>* d_bias;
    gpuErrchk(cudaMalloc((void**)&d_bias, sizeof(DimmedGaussGridGPU<1>)));
    gpuErrchk(cudaMemcpy(d_bias, bias, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    add_value_integral_kernel<1><<<1, grid_dims>>>(buffer, target, d_bias);
    gpuErrchk(cudaFree(d_bias));
    gpuErrchk(cudaDeviceSynchronize());
    return;
  case 2:
    DimmedGaussGridGPU<2>* d_bias2;
    gpuErrchk(cudaMalloc((void**)&d_bias2, sizeof(DimmedGaussGridGPU<2>)));
    gpuErrchk(cudaMemcpy(d_bias2, bias, sizeof(DimmedGaussGridGPU<2>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    add_value_integral_kernel<2><<<1, grid_dims>>>(buffer, target, d_bias2);
    gpuErrchk(cudaFree(d_bias2));
    gpuErrchk(cudaDeviceSynchronize());
    return;
  case 3:
    DimmedGaussGridGPU<3>* d_bias3;
    gpuErrchk(cudaMalloc((void**)&d_bias3, sizeof(DimmedGaussGridGPU<3>)));
    gpuErrchk(cudaMemcpy(d_bias3, bias, sizeof(DimmedGaussGridGPU<3>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    add_value_integral_kernel<3><<<1, grid_dims>>>(buffer, target, d_bias3);
    gpuErrchk(cudaFree(d_bias3));
    gpuErrchk(cudaDeviceSynchronize());
    return;
  }
  return;
}
