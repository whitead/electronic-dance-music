#include "edm_bias_gpu.cuh"
#include "grid_gpu.cuh"
#include "gaussian_grid_gpu.cuh"
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


using namespace EDM_Kernels;


EDM::EDMBiasGPU::EDMBiasGPU(const std::string& input_filename) : EDMBias(input_filename) {
  printf("Called EDMBiasGPU constructor...\n");
  gpuErrchk(cudaMallocManaged(&send_buffer_, sizeof(edm_data_t) * BIAS_BUFFER_DBLS));
  for(int i = 0; i < BIAS_BUFFER_DBLS; i++){
    send_buffer_[i] = 0;//CUDA wants this..?
  }
  read_input(input_filename);
}


EDM::EDMBiasGPU::~EDMBiasGPU() {
  gpuErrchk(cudaDeviceSynchronize());
  /* 
   * target_, bias_dx_, and cv_hist_ are all delete'd by the superclass, EDMBias
   */
  if(mpi_neighbors_ != NULL){
    gpuErrchk(cudaFree(mpi_neighbors_));
    mpi_neighbors_ = NULL;
  }
    
  if(bias_dx_ != NULL){
    gpuErrchk(cudaFree(bias_dx_));
    bias_dx_ = NULL;
  }
    
  if(bias_sigma_ != NULL){
    gpuErrchk(cudaFree(bias_sigma_));
    bias_sigma_ = NULL;
  }
    
  if(min_ != NULL){
    gpuErrchk(cudaFree(min_));
    min_ = NULL;
  }
    
  if(max_ != NULL){
    gpuErrchk(cudaFree(max_));
    max_ = NULL;
  }
    
  if(b_periodic_boundary_ != NULL){
    gpuErrchk(cudaFree(b_periodic_boundary_));
    b_periodic_boundary_ = NULL;
  }
    

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
    hill_output_.open(cleaned_filename.c_str());    
  } else {
    string hfilename("HILLS");
    string cleaned_filename = clean_string(hfilename, 1);
    hill_output_.open(cleaned_filename.c_str());

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
      add_hill(&positions[i][0], runiform[i]);
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
  printf("post_add_hill was called. steps_ = %d\n", steps_);
}


void EDM::EDMBiasGPU::queue_add_hill(const edm_data_t* position, edm_data_t this_h){
  //use the same buffer system but there's only one cause it's just GPU
  size_t i;
  for(i = 0; i < dim_; i++)
    send_buffer_[buffer_i_ * (dim_+ 1) + i] = position[i];
  send_buffer_[buffer_i_ * (dim_ + 1) + i] = this_h;
  buffer_i_++;
  
  //do we need to flush?
  if(buffer_i_ >= BIAS_BUFFER_SIZE)
    temp_hill_cum_ += flush_buffers(0); //flush and we don't know if we're synched
  
}

edm_data_t EDM::EDMBiasGPU::flush_buffers(int synched) {
  edm_data_t bias_added = 0;
  //flush the buffer. only one with GPU version for now...
  //TODO: make this a kernel call!
  bias_added += do_add_hills(send_buffer_, buffer_i_, ADD_HILL);

  //reset buffer count
  buffer_i_ = 0;

  return bias_added;
}

edm_data_t EDM::EDMBiasGPU::do_add_hills(const edm_data_t* buffer, const size_t hill_number, char hill_type){
  /*TODO: REPLACE this with a properly-calculated kernel call such that we can recover
  ** the bias_added but only call kernel *once*...
  ** Note: see edm_gpu_test.cu:1377 for how that's called.
  */
  edm_data_t bias_added = 0;
  printf("edm_bias_gpu.cu:307 Called do_add_hills.\n");
  size_t i, j;
  int minisize = bias_->get_minisize_total();//this works
  printf("edm_bias_gpu.cu:309 minisize set to %d, with hill number = %zd\n", minisize, hill_number);
  dim3 grid_dims(minisize, hill_number);
  edm_data_t* d_bias_added;
  gpuErrchk(cudaMallocManaged((void**)&d_bias_added, minisize * hill_number * sizeof(edm_data_t)));
  for(i = 0; i < minisize * hill_number; i++){
    d_bias_added[i] = 0;
  }
  //TODO: FIX THIS SEGFAULT!
  launch_add_value_integral_kernel(dim_, buffer, d_bias_added, bias_, grid_dims);//this launches kernel.

//  for(i = 0; i < hill_number; i++){
//    for(j = 0; j < minisize; j++){
//      bias_added += d_bias_added[i*minisize + j];//sum over each mini-grid
//    }
//    output_hill(&buffer[i * (dim_ + 1)], buffer[i * (dim_ + 1) + dim_], bias_added, hill_type);
//  }
  hills_added_ += hill_number;

  return bias_added;
}

/*have to use another switch-based function to dynamically launch 
 *a kernel without prior knowledge of what dimension we're using.
 */
void EDM::EDMBiasGPU::launch_add_value_integral_kernel(int dim, const edm_data_t* buffer, edm_data_t* target, Grid* bias, dim3 grid_dims){
  printf("Called launch_add_value_integral kernel with dimension %d!\n", dim);
  switch(dim){
  case 1:
    DimmedGaussGridGPU<1>* d_bias;
    printf("Trying to copy a 1D grid to GPU...\n");
    gpuErrchk(cudaMalloc((void**)&d_bias, sizeof(DimmedGaussGridGPU<1>)));
    gpuErrchk(cudaMemcpy(d_bias, bias, sizeof(DimmedGaussGridGPU<1>), cudaMemcpyHostToDevice));
    printf("Successfully copied the 1D grid to GPU.\n");
    gpuErrchk(cudaDeviceSynchronize());
    printf("launching kernel with grid_dims x = %d, y = %d...\n", grid_dims.x, grid_dims.y);
    add_value_integral_kernel<1><<<1, grid_dims>>>(buffer, target, d_bias);
    printf("Now exiting call to launch_add_value_integral_kernel...\n");
    return;
  case 2:
    DimmedGaussGridGPU<2>* d_bias2;
    gpuErrchk(cudaMalloc((void**)&d_bias2, sizeof(DimmedGaussGridGPU<2>)));
    gpuErrchk(cudaMemcpy(d_bias2, bias, sizeof(DimmedGaussGridGPU<2>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    add_value_integral_kernel<2><<<1, grid_dims>>>(buffer, target, d_bias2);
    printf("Now exiting call to launch_add_value_integral_kernel...\n");
    return;
  case 3:
    DimmedGaussGridGPU<3>* d_bias3;
    gpuErrchk(cudaMalloc((void**)&d_bias3, sizeof(DimmedGaussGridGPU<3>)));
    gpuErrchk(cudaMemcpy(d_bias3, bias, sizeof(DimmedGaussGridGPU<3>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    add_value_integral_kernel<3><<<1, grid_dims>>>(buffer, target, d_bias3);
    printf("Now exiting call to launch_add_value_integral_kernel...\n");
    return;
  }
  printf("SHOULD NOT BE HERE!!!\n");
  return;
}
