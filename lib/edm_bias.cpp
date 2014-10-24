#include "edm_bias.h"

#include <cmath>
#include <iterator>
#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include "mpi.h"
 

//Some stuff for reading in files quickly 
namespace std {
  istream& operator >> (istream& is, pair<string, string>& ps) {
      is >> ps.first;
      std::getline (is,ps.second);
      return is;
  }

  ostream& operator << (ostream& os, const pair<const string, string>& ps)
  {
    return os << "\"" << ps.first << "\": \"" << ps.second << "\"";
  }

}


EDMBias::EDMBias(const std::string& input_filename) : b_tempering_(0), global_tempering_(0), hill_density_(-1),
						      cum_bias_(0), b_outofbounds_(0), target_(NULL), bias_(NULL), mask_(NULL){
  
  //read input file
  read_input(input_filename);
  
}

EDMBias::~EDMBias() {
  if(target_ != NULL)
    delete target_;
  if(bias_ != NULL)
    delete bias_;
}


void EDMBias::subdivide(const double sublo[3], const double subhi[3], const int b_periodic[3]) {

  int no_period[] = {0, 0, 0};
  double min[3];
  double max[3];
  size_t i, j;

  int bounds_flag = 1;
  for(i = 0; i < dim_; i++) {
    min[i] = fmax(sublo[i] - bias_sigma_[i] * 6, min_[i]);
    max[i] = fmin(subhi[i] + bias_sigma_[i] * 6, max_[i]);
    //check if we'll always be out of bounds
    bounds_flag &= (min[i] >= max_[i] || max[i] <= min_[i]);

  }

  
  delete bias_;
  bias_ = make_gauss_grid(dim_, min, max, bias_dx_, no_period, 1, bias_sigma_);
  bias_->set_boundary(min_, max_, b_periodic);

  if(bounds_flag) {
    //we do this after so that we have a grid to at least write out
    b_outofbounds_ = 1;
    return;
  }

  /* I've decided not to subdivide the target for now
  Grid* new_target = make_grid(dim_, sublo, subhi, target_->get_dx(), no_period, 0, 1);
  
  size_t index[3];
  double x[3];
  for(i = 0; i < new_target->get_grid_size(); i++) {
    new_target->one2multi(i, index);
    for(j = 0; j < dim_; j++) {
      x[j] = index[j] * new_target->get_dx()[j] + new_target->get_min()[j];
    }
    new_target->get_grid()[i] = target_->get_value(x);
  }
  delete target_;
  target_ = new_target;
  */

  //get volume
  double other_vol = 0;
  double vol = bias_->get_volume();
  total_volume_ = 0;
  MPI_Allreduce(&vol, &other_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  total_volume_ += other_vol;

}

void EDMBias::write_bias(const std::string& output) const {
  bias_->multi_write(output, min_, max_);
}

void EDMBias::setup(double temperature, double boltzmann_factor) {

  temperature_ = temperature;
  boltzmann_factor_ = boltzmann_factor;

}

void EDMBias::update_forces(int nlocal, const double* const* positions, double** forces) const {
  update_forces(nlocal, positions, forces, -1);
}


void EDMBias::update_forces(int nlocal, const double* const* positions, double** forces, int apply_mask) const {

  //are we active?
  if(b_outofbounds_)
    return;

  //simply perform table look-ups of the positions to get the forces
  int i,j;
  double der[3] = {0, 0, 0};
  for(i = 0; i < nlocal; i++) {
    if(apply_mask > 0 && mask_[i] & apply_mask) {
      bias_->get_value_deriv(&positions[i][0], der);
      for(j = 0; j < 0; j++)
	forces[i][j] -= der[j];
    }
  }

}

void EDMBias::add_hills(int nlocal, const double* const* positions, const double* runiform) {
  add_hills(nlocal, positions, runiform, -1);
}

void EDMBias::add_hills(int nlocal, const double* const* positions, const double* runiform, int apply_mask) {

  //are we active?
  if(b_outofbounds_)
    return;

  int i;

  //get current hill height
  double bias_added = 0;
  double h = hill_prefactor_;  
  if(global_tempering_ > 0)
    if(cum_bias_ / total_volume_ >= global_tempering_)
      h *= exp(-(cum_bias_ / total_volume_ - global_tempering_) / ((bias_factor_ - 1) * temperature_ * boltzmann_factor_));                   
  double this_h;

  //count how many atoms we have in bounds
  int natoms = 0;
   for(i = 0; i < nlocal; i++)
     if(apply_mask > 0 && mask_[i] & apply_mask)
       if(bias_->in_bounds(&positions[i][0]))
	 natoms++;


  for(i = 0; i < nlocal; i++)  {   

    if(apply_mask > 0 && mask_[i] & apply_mask) {
      //actually add hills -> stochastic
      if(hill_density_ > 0 && runiform[i] < hill_density_ / natoms) {    
	this_h = h; 
	this_h /= exp(target_->get_value(&positions[i][0])); // add target
	if(b_tempering_ && global_tempering_ < 0) //do tempering if local tempering (well) is being used
	  this_h *= exp(-bias_->get_value(&positions[i][0]) / ((bias_factor_ - 1) * temperature_ * boltzmann_factor_));
	//finally clamp bias
	this_h = fmin(this_h, BIAS_CLAMP * boltzmann_factor_);
	bias_added += bias_->add_gaussian(&positions[i][0], this_h);
      }
    }
  }

  update_height(bias_added);

}

void EDMBias::update_height(double bias_added) {
  double other_bias = 0;
  MPI_Allreduce(&bias_added, &other_bias, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  cum_bias_ += other_bias;
		
}

int extract_double(const std::string& key, std::map<std::string, std::string> map, int required, double* result) {

  if(map.find(key) != map.end()) {
    *result = atof(map.at(key).c_str());
    if(*result == 0.0) {
      std::cerr << "Invalid value found for " << key << " (" << result << ")" << std::endl;    
      return 0;
    }
    return 1;
    
  } else{
    if(required)
      std::cerr << "Could not find key " << key << " (" << result << ")" << std::endl;    
  }

  return 0;
  
}

int extract_double_array(const std::string& key, std::map<std::string, std::string> map, int required, double* result, int length) {

  if(map.find(key) != map.end()) {
    std::istringstream is(map.at(key));
    for(int i = 0; i < length; i++)
      is >> result[i];
    return 1;    
  } else{
    if(required)
      std::cerr << "Could not find key " << key << " (" << result << ")" << std::endl;
  }

  return 0;
  
}

int extract_int(const std::string& key, std::map<std::string, std::string> map, int required, int* result) {

  if(map.find(key) != map.end()) {
    *result = atoi(map.at(key).c_str());    
    return 1;
  } else{
    if(required)
      std::cerr << "Could not find key " << key << " (" << result << ")" << std::endl;    
  }
  return 0;
  
}


void EDMBias::set_mask(const int* mask) {
  mask_ = mask;
}

int EDMBias::read_input(const std::string& input_filename){ 

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
    if(!extract_double("bias_factor", parsed_input, 1,&bias_factor_))
      return 0;
    extract_double("global_tempering", parsed_input, 0,&global_tempering_);    
  }
  
  if(!extract_double("hill_prefactor", parsed_input, 1, &hill_prefactor_))
    return 0;
  extract_double("hill_density", parsed_input, 0, &hill_density_);
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
  bias_dx_ = (double*) malloc(sizeof(double) * dim_);
  bias_sigma_ = (double*) malloc(sizeof(double) * dim_);
  min_ = (double*) malloc(sizeof(double) * dim_);
  max_ = (double*) malloc(sizeof(double) * dim_);
  if(!extract_double_array("bias_spacing", parsed_input, 1, bias_dx_, dim_))
    return 0;
  if(!extract_double_array("bias_sigma", parsed_input, 1, bias_sigma_, dim_))
    return 0;
  if(!extract_double_array("box_low", parsed_input, 1, min_, dim_))
    return 0;
  if(!extract_double_array("box_high", parsed_input, 1, max_, dim_))
    return 0;

  //get target
  if(parsed_input.find("target_filename") == parsed_input.end()) {
    cerr << "Must specify target" << endl;
    return 0;
  } else {
    string& tfilename = parsed_input.at("target_filename");
    //remove surrounding whitespace 
    size_t found = tfilename.find_first_not_of(" \t");
    if (found != string::npos)
      tfilename = tfilename.substr(found);
    target_ = read_grid(dim_, tfilename);
  }
 
  return 1;
}
