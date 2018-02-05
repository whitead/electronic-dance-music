#include "edm_bias_gpu.cuh"
#include "grid_gpu.cuh"
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





EDM::EDMBiasGPU::EDMBiasGPU(const std::string& input_filename) : EDMBias(input_filename) {
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

