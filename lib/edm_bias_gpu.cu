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





EDM::EDMBiasGPU::EDMBiasGPU(const std::string& input_filename) : EDMBias(input_filename) {
  read_input(input_filename);
}


EDM::EDMBiasGPU::~EDMBiasGPU() {
  //nothing yet
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

  //  copy(parsed_input.begin(), parsed_input.end(), ostream_iterator<pair<string,string> >(cout, "\n"));
 
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
    target_ = read_grid(dim_, cleaned_filename, 0); //read grid, do not use interpolation
    expected_target_ = target_->expected_bias();
    std::cout << "Expected Target is " << expected_target_ << std::endl;
  }

  if(parsed_input.find("initial_bias_filename") == parsed_input.end()) {
    initial_bias_ = NULL;
  } else {
    string ibfilename = parsed_input.at("initial_bias_filename");
    string cleaned_filename = clean_string(ibfilename, 0);
    initial_bias_ = read_grid(dim_, cleaned_filename, 1); //read grid, do use interpolation
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

