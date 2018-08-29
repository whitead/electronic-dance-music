#include "edm_bias.h"
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
#include "mpi.h"

#ifdef EDM_MPI_DEBUG
#include "unistd.h"
#endif

#ifdef USE_DOUBLES
#define MPI_EDM_DATA_T MPI_DOUBLE
#else
#define MPI_EDM_DATA_T MPI_FLOAT
#endif//USE_DOUBLES

namespace EDM{

  int extract_edm_data_t(const std::string& key, std::map<std::string, std::string> map, int required, edm_data_t* result) {

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

  int extract_edm_data_t_array(const std::string& key, std::map<std::string, std::string> map, int required, edm_data_t* result, int length) {

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
}

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


EDM::EDMBias::EDMBias(const std::string& input_filename) : b_tempering_(0), 
							   b_targeting_(0), 
							   mpi_rank_(0),
							   mpi_size_(0),
							   global_tempering_(0), 
							   temperature_(-1.0),
							   hill_density_(-1),
							   cum_bias_(0), 
							   b_outofbounds_(0), 
							   target_(NULL), 
							   bias_(NULL), 
							   mask_(NULL),
							   mpi_neighbor_count_(0),
							   mpi_neighbors_(NULL),
							   buffer_i_(0),
							   temp_hill_cum_(-1),
							   steps_(0),		   
							   temp_hill_prefactor_(-1),
							   est_hill_count_(0),
							   cv_hist_(NULL),
							   min_(NULL),
							   max_(NULL),
							   bias_dx_(NULL),
							   bias_sigma_(NULL),
							   b_periodic_boundary_(NULL){
  
  //assign rank
#ifndef EDM_SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size_);
#endif
  for(int i = 0; i < BIAS_BUFFER_DBLS; i++){
    send_buffer_[i] = 0;
    receive_buffer_[i] = 0;
  }
  //read input file
  read_input(input_filename);
}
  
EDM::EDMBias::~EDMBias() {
  if(target_ != NULL)
    delete target_;
  if(bias_ != NULL)
    delete bias_;
  if(cv_hist_ != NULL)
    delete cv_hist_;
  if(mpi_neighbors_ != NULL)
    free(mpi_neighbors_);
  if(bias_dx_ != NULL)
    free(bias_dx_);
  if(bias_sigma_ != NULL)
    free(bias_sigma_);
  if(min_ != NULL)
    free(min_);
  if(max_ != NULL)
    free(max_);
  if(b_periodic_boundary_ != NULL)
    free(b_periodic_boundary_);

}
  

//need to also pass the the box size and its periodicity so we can
//infer if the given boundary extends across the entire system. That
//determines the acutal b_periodic

void EDM::EDMBias::subdivide(const edm_data_t sublo[3], 
			     const edm_data_t subhi[3], 
			     const edm_data_t boxlo[3],
			     const edm_data_t boxhi[3],
			     const int b_periodic[3],
			     const edm_data_t skin[3]) {

#ifdef EDM_MPI_DEBUG
  if(mpi_rank_ == 2) {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (43 == i)
      sleep(5);
  }
  

#endif


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

  bias_ = make_gauss_grid(dim_, min, max, bias_dx_, grid_period, INTERPOLATE, bias_sigma_);
  //create histogram with no interpolation/no derivatives for tracking CV
  cv_hist_ = make_grid(dim_, min, max, bias_sigma_, grid_period, 0, 0);
    
  bias_->set_boundary(min_, max_, b_periodic_boundary_);
  if(initial_bias_ != NULL)
    bias_->add(initial_bias_, 1.0, 0.0);
  

#ifndef EDM_SERIAL
  infer_neighbors(b_periodic, skin);

  //make hill density a per-system measurement not per replica
  //make hill prefactor also lower so bias per timestep is not a function of replica number
  if(hill_density_ > 0)  {
    hill_density_ /=  mpi_size_;
    hill_prefactor_ /= mpi_size_;
    if(hill_density_ == 0)
      hill_density_  = 1;
  }

  //We have two comm cases, global and neighbors. In global, we build
  //a broadcast tree so that the number of comms is logrithmic in the
  //number of comms. That is with broadcast. If instead it's more
  //efficient to use neighbor communication, then we need to sort our neighbors
  if(mpi_neighbor_count_ < log(mpi_size_)) {
    //  if(1)
    sort_neighbors();
    std::cout << "Using P2P with " << mpi_neighbor_count_ << " neighbors" << std::endl;
  } else{
    mpi_neighbor_count_ = mpi_size_ - 1; //just communicate with all
    std::cout << "Using broadcast with " << mpi_neighbor_count_ << " neighbors" << std::endl;
  }


  
#endif
  
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

#ifndef EDM_SERIAL
  MPI_Allreduce(&vol, &other_vol, 1, MPI_EDM_DATA_T, MPI_SUM, MPI_COMM_WORLD);
#else
  other_vol = vol;
#endif
  total_volume_ += other_vol;

}

void EDM::EDMBias::write_bias(const std::string& output) const {
#ifndef EDM_SERIAL
  bias_->multi_write(output);
#ifdef EDM_MPI_DEBUG
  std::ostringstream oss;
  oss << output << "_" << mpi_rank_;
  bias_->write(oss.str());
#endif //EDM_DEBUG
#else //SERIAL TEST
  bias_->write(output);
#endif
}

void EDM::EDMBias::write_histogram() const {
#ifndef EDM_SERIAL
  cv_hist_->multi_write(hist_output_, min_, max_, b_periodic_boundary_, 0);
#ifdef EDM_MPI_DEBUG
  std::ostringstream oss;
  oss << hist_output_ << "_" << mpi_rank_;
  cv_hist_->write(oss.str());
#endif //EDM_DEBUG
#else //SERIAL TEST
  cv_hist_->write(hist_output_);
#endif
}

void EDM::EDMBias::clear_histogram() {
  cv_hist_->clear();
}

void EDM::EDMBias::write_lammps_table(const std::string& output) const {
#ifndef EDM_SERIAL
  bias_->lammps_multi_write(output);
#else
  bias_->write(output);
#endif//SERIAL TEST  


}

void EDM::EDMBias::setup(edm_data_t temperature, edm_data_t boltzmann_constant) {

  temperature_ = temperature;
  boltzmann_factor_ = boltzmann_constant * temperature;

}

edm_data_t EDM::EDMBias::update_forces(int nlocal, const edm_data_t* const* positions, edm_data_t** forces) const {
  return update_forces(nlocal, positions, forces, -1);
}


edm_data_t EDM::EDMBias::update_forces(int nlocal, const edm_data_t* const* positions, edm_data_t** forces, int apply_mask) const {

  //are we active?
  if(b_outofbounds_)
    return 0.0;
  
  
  //simply perform table look-ups of the positions to get the forces
  int i,j;
  edm_data_t der[3] = {0, 0, 0};
  edm_data_t energy = 0;
  for(i = 0; i < nlocal; i++) {
    if(apply_mask < 0 || mask_[i] & apply_mask) {
      energy += bias_->get_value_deriv(&positions[i][0], der);
      for(j = 0; j < dim_; j++)
	forces[i][j] -= der[j];
    }
  }
  return energy;
}

edm_data_t EDM::EDMBias::update_force(const edm_data_t* positions, edm_data_t* forces) const {

  //are we active?
  if(b_outofbounds_)
    return 0.0;
  
  
  //simply perform table look-ups of the positions to get the forces
  int i;
  edm_data_t der[3] = {0, 0, 0};
  edm_data_t energy = bias_->get_value_deriv(positions, der);
  for(i = 0; i < dim_; i++)
    forces[i] -= der[i];
  return energy;
}


void EDM::EDMBias::add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform) {
  add_hills(nlocal, positions, runiform, -1);
}

void EDM::EDMBias::add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform, int apply_mask) {

  int i;
  pre_add_hill(nlocal);
  for(i = 0; i < nlocal; i++) {
    if(apply_mask < 0 || apply_mask & mask_[i])
      add_hill(&positions[i][0], runiform[i]);
  }
  post_add_hill();

}

void EDM::EDMBias::pre_add_hill(int est_hill_count) {

  //are we active?
  if(!b_outofbounds_) {

    est_hill_count_ = est_hill_count;
    temp_hill_prefactor_ = hill_prefactor_;

    //get current hill height
    if(global_tempering_ > 0)
      if(cum_bias_ / total_volume_ >= global_tempering_)
	temp_hill_prefactor_ *= exp(-(cum_bias_ / total_volume_ - 
				      global_tempering_) / 
				    (global_tempering_ * (bias_factor_ - 1) * boltzmann_factor_));                   

    temp_hill_cum_ = 0;
    hills_added_ = 0;

    //we can only skip entire rounds, otherwise we begin to bias our sampling.
    //Are we skipping an entire round?
  }

}

edm_data_t EDM::EDMBias::do_add_hills(const edm_data_t* buffer, const size_t hill_number, char hill_type){
  edm_data_t bias_added = 0;
  
  size_t i;
  for(i = 0; i < hill_number; i++){
    //(dim_+1)*i here b/c of the height terms
    bias_added += bias_->add_value(&buffer[i * (dim_ + 1)], buffer[i * (dim_ + 1) + dim_]);    
    hills_added_++;
    output_hill(&buffer[i * (dim_ + 1)], buffer[i * (dim_ + 1) + dim_], bias_added, hill_type);
  }

  return bias_added;
}

void EDM::EDMBias::queue_add_hill(const edm_data_t* position, edm_data_t this_h){

  size_t i;
  for(i = 0; i < dim_; i++)
    send_buffer_[buffer_i_ * (dim_+ 1) + i] = position[i];
  send_buffer_[buffer_i_ * (dim_ + 1) + i] = this_h;
  buffer_i_++;
  
  //do we need to flush?
  if(buffer_i_ >= BIAS_BUFFER_SIZE -1 )
    temp_hill_cum_ += flush_buffers(0); //flush and we don't know if we're synched
  
}


void EDM::EDMBias::add_hill(const edm_data_t* position, edm_data_t runiform) {

  if(temp_hill_prefactor_ < 0)
    edm_error("Must call pre_add_hill before add_hill", "edm_bias.cpp:add_hill");
  
  edm_data_t this_h = temp_hill_prefactor_;

  //are we active?
  if(!b_outofbounds_) {

    //actually add hills -> stochastic
    if(hill_density_ < 0 || runiform < hill_density_ / est_hill_count_) {    

      if(b_targeting_)
	this_h *= exp(target_->get_value(position) - expected_target_); // add target and compensate for expected target
      if(b_tempering_ && global_tempering_ < 0) //do tempering if local tempering (well) is being used
	this_h *= exp(-bias_->get_value(position) / 
		      ((bias_factor_ - 1) * boltzmann_factor_));

      //compensate for number of hills, if necessary
      if(hill_density_ < 0)
	this_h /= est_hill_count_;
      else
	this_h /= hill_density_;

      //finally clamp bias
      this_h = fmin(this_h, BIAS_CLAMP);
      queue_add_hill(position, this_h);
    }
  }
}

void EDM::EDMBias::post_add_hill() {

  if(temp_hill_cum_ < 0) {
    //error must call pre_add_hill before post_add_hill
  }

  temp_hill_cum_ += flush_buffers(0); //flush, but we don't know if we're synched

  //some processors may have had to flush more than once if there are
  //lots of hills, continue waiting for them until we all agree no more flush
  while(check_for_flush())
    temp_hill_cum_ += flush_buffers(1); //flush and we are synched
  
  update_height(temp_hill_cum_);

  temp_hill_cum_ = -1;
  temp_hill_prefactor_ = -1;
  steps_++;
}
 

void EDM::EDMBias::output_hill(const edm_data_t* position, edm_data_t height, edm_data_t bias_added, char type) {
  
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
    cv_hist_->add_value(position, 1);
  } else if(type == ADD_UNDO_HILL ||
	    type == BUFF_UNDO_HILL) {
    //undo histogram
    cv_hist_->add_value(position, -1);
  }
   
}

int EDM::EDMBias::check_for_flush() {
  
  if(mpi_neighbor_count_ > 0) {
    
    int my_flush = 0;
    int do_flush;
    MPI_Allreduce(&my_flush, &do_flush, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(do_flush)
      return 1;
    
  }
  
  return 0;

}

edm_data_t EDM::EDMBias::flush_buffers(int synched) {
  edm_data_t bias_added = 0;
  
  //flush our own buffer first
  bias_added += do_add_hills(send_buffer_, buffer_i_, ADD_HILL);

  if(mpi_neighbor_count_ > 0) {
    
    //notify all that we're going to flush
    if(!synched) {
      int my_flush = 1;
      int do_flush;
      MPI_Allreduce(&my_flush, &do_flush, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    
    size_t i;
    unsigned int buffer_j;
    MPI_Request srequest1, srequest2;

    if(mpi_neighbor_count_ == mpi_size_ - 1) {
      for(i = 0; i < mpi_size_; i++) {
	if(mpi_rank_ == i) {
	  MPI_Bcast(&buffer_i_, 1, MPI_UNSIGNED, i, MPI_COMM_WORLD);
	  MPI_Bcast(send_buffer_, buffer_i_ * (dim_ + 1), MPI_EDM_DATA_T, i, MPI_COMM_WORLD);
	} else {
	  MPI_Bcast(&buffer_j, 1, MPI_UNSIGNED, i, MPI_COMM_WORLD);
	  MPI_Bcast(receive_buffer_, buffer_j * (dim_ + 1), MPI_EDM_DATA_T, i, MPI_COMM_WORLD);
	  bias_added += do_add_hills(receive_buffer_, buffer_j, NEIGH_HILL);	  
	}
      }
    } else {
      
      for(i = 0; i < mpi_neighbor_count_; i++) {

	if(mpi_neighbors_[i] == NO_COMM_PARTNER)
	  continue;

	//async send, since we don't care about the buffer
	MPI_Isend(&buffer_i_, 1, MPI_UNSIGNED, 
		  mpi_neighbors_[i], i, MPI_COMM_WORLD, 
		  &srequest1);

	MPI_Isend(send_buffer_, buffer_i_ * (dim_ + 1), 
		  MPI_EDM_DATA_T, mpi_neighbors_[i], 
		  i, MPI_COMM_WORLD, &srequest2);

	//do sync receive, because we need it to continue
	MPI_Recv(&buffer_j, 1, MPI_UNSIGNED, 
		 mpi_neighbors_[i], i, 
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Recv(receive_buffer_, buffer_j * (dim_ + 1), 
		 MPI_EDM_DATA_T, mpi_neighbors_[i], 
		 i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

	bias_added += do_add_hills(receive_buffer_, buffer_j, NEIGH_HILL);	  	

	//Technically, I don't need to do this here but 
	//it's more convienent than having a bunch of outstanding requests
	MPI_Wait(&srequest1, MPI_STATUS_IGNORE);
	MPI_Wait(&srequest2,  MPI_STATUS_IGNORE);
      }
    }

  }

  //reset buffer, even if we don't have other MPI
  buffer_i_ = 0;

  return bias_added;
}

void EDM::EDMBias::infer_neighbors(const int* b_periodic, const edm_data_t* skin) {

  //now the hard part, we need to infer the domain decomposition topology
  size_t i,j;
  edm_data_t* bounds = (edm_data_t*) malloc(sizeof(edm_data_t) * dim_ * 2);
  int dim_overlap; //if somethig overlaps in all dimensions
   

  if(mpi_neighbors_ != NULL)
    free(mpi_neighbors_);

  mpi_neighbors_ = (int*) malloc(sizeof(int) * mpi_size_);
  for(i = 0; i < mpi_size_; i++) 
    mpi_neighbors_[i] = NO_COMM_PARTNER;

  for(i = 0; i < mpi_size_; i++) {   //for each rank 

    if(mpi_rank_ == i) {//it's my turn to broadcast my boundaries

      for(j = 0; j < dim_; j++) {//pack up my bounds
	bounds[j*2] = bias_->get_min()[j];
	bounds[j*2 + 1] = bias_->get_max()[j];
      }
      MPI_Bcast(bounds, dim_ * 2, MPI_EDM_DATA_T, i, MPI_COMM_WORLD);

    }  else {
      MPI_Bcast(bounds, dim_ * 2, MPI_EDM_DATA_T, i, MPI_COMM_WORLD);
       
      //check if this could be a neighbor
      dim_overlap = 0;
      for(j = 0; j < dim_; j++) {
	//check if their min is within our bounds
	if(bounds[j * 2] < (bias_->get_max()[j] + GAUSS_SUPPORT * bias_sigma_[j]) &&
	   bounds[j * 2] > (bias_->get_min()[j] - GAUSS_SUPPORT * bias_sigma_[j])) {
	  dim_overlap++;
	  continue;
	  //or if their max is wihtin our bounds
	} else if(bounds[j * 2 + 1] < (bias_->get_max()[j] + GAUSS_SUPPORT * bias_sigma_[j]) &&
		  bounds[j * 2 + 1] > (bias_->get_min()[j] - GAUSS_SUPPORT * bias_sigma_[j])) {
	  dim_overlap++;
	  continue;
	}
	//those were the easy cases, now wrapping
	if(b_periodic[j]) {

	  if(fabs(bias_->get_min()[j] - min_[j] + skin[j]) < GAUSS_SUPPORT * bias_sigma_[j] && // I am at left
	     fabs(bounds[j * 2 + 1] - max_[j] - bias_dx_[j] - skin[j]) < GAUSS_SUPPORT * bias_sigma_[j]) { //other is at right
	    dim_overlap++;
	    continue;
	  } else if(fabs(bias_->get_max()[j] - max_[j] - bias_dx_[j] - skin[j]) < GAUSS_SUPPORT * bias_sigma_[j] && //or I am at right
		    fabs(bounds[j * 2] - min_[j] + skin[j]) < GAUSS_SUPPORT * bias_sigma_[j]) {//other is at left
	    dim_overlap++;
	    continue;
	  }
	}
      }
      if(dim_overlap == dim_) {
	mpi_neighbors_[mpi_neighbor_count_] = i;
	mpi_neighbor_count_++;
      }
    }
  }

  //print out the unsorted neighbors
  for(i = 0; i < mpi_size_; i++) {
    if(mpi_rank_ == i) {
      std::cout << "Neighobrs of " << i << " [";
      for(j = 0; j < dim_; j++)
	std::cout << bias_->get_min()[j] << ", ";
      std::cout << "-> ";
      for(j = 0; j < dim_; j++)
	std::cout << bias_->get_max()[j] << ", ";
      std::cout << "] == ";

      for(j = 0; j < mpi_neighbor_count_; j++){
	std::cout << mpi_neighbors_[j] << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}


/** This method will take the unordered list of neighbors and create sorted versions
 * so that there will be no communication blocks
 */
void EDM::EDMBias::sort_neighbors() {

  int* unsorted_neighbors; 
  unsigned int* unsorted_counts;
  int* sorted_neighbors; 
  unsigned int* sorted_counts;//this is include NO_COMM_PARTNER neighbors, so is generally higher
  unsigned int i,j,k,l;
  int* b_paired;
  int flag;

  //only execute on head node, because it's too much comm to try to parallelize it
  if(mpi_rank_ == 0) {
     
    sorted_neighbors = (int*) malloc(sizeof(int) * mpi_size_ * mpi_size_);
    unsorted_neighbors = (int*) malloc(sizeof(int) * mpi_size_ * mpi_size_);
    sorted_counts = (unsigned int*) malloc(sizeof(unsigned int) * mpi_size_);
    unsorted_counts = (unsigned int*) malloc(sizeof(unsigned int) * mpi_size_);
    b_paired = (int*) malloc(sizeof(int) * mpi_size_);
     
    for(i = 0; i < mpi_size_; i++)
      sorted_counts[i] = 0;

  }

  MPI_Gather(mpi_neighbors_, mpi_size_, MPI_INT, unsorted_neighbors, mpi_size_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&mpi_neighbor_count_, 1, MPI_UNSIGNED, unsorted_counts, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if(mpi_rank_ == 0) {
     
    for(i = 0; i < mpi_size_; i++) {//for each round of communication with neighbors, at most size size
      //clear paired flags
      for(j = 0; j < mpi_size_; j++)
	b_paired[j] = 0;

      for(j = 0; j < mpi_size_; j++) {//for each node
	if(!b_paired[j]) {

	  //assume we won't find a neighbor
	  sorted_neighbors[j * mpi_size_ + sorted_counts[j]] = NO_COMM_PARTNER;
	   
	  //iterate through my neighbor list
	  for(k = 0; k < mpi_size_ && unsorted_neighbors[j * mpi_size_ + k] != NO_COMM_PARTNER; k++) {
	    //NO_COMM_PARTNER happens to mark the end of the array

	    if(!b_paired[unsorted_neighbors[j * mpi_size_ + k]]) { //found an unpaired neighbor!

	      //make sure we haven't already paired with it once before
	      flag = 0;
	      for(l = 0; l < sorted_counts[j]; l++) {
		if(sorted_neighbors[j * mpi_size_ + l] == unsorted_neighbors[j * mpi_size_ + k]) {
		  flag = 1;
		  break;
		}
	      }

	      if(!flag) {

		l = unsorted_neighbors[j * mpi_size_ + k];

		sorted_neighbors[j * mpi_size_ + sorted_counts[j]] = l;
		unsorted_counts[j]--;

		sorted_neighbors[l * mpi_size_ + sorted_counts[l]] = j;
		b_paired[l] = 1;
		sorted_counts[l]++;
		unsorted_counts[l]--;
		break;
	      }
	    }
	  }
	   
	  sorted_counts[j]++;	   
	  b_paired[j] = 1; //we are either paired or know we're exlcuded
	   
	}
      }


#ifdef EDM_MPI_DEBUG
      for(j = 0; j < mpi_size_; j++) {
	std::cout << j << ": [";
	for(k = 0; k < sorted_counts[j]; k++)
	  std::cout << sorted_neighbors[j * mpi_size_ + k] << ", ";
	std::cout << "] <--> [";
	for(k = 0; k < unsorted_counts[j]; k++)
	  std::cout << unsorted_neighbors[j * mpi_size_ + k] << ", ";
	std::cout << "]" << std::endl;
	 
      }
#endif// EDM_MPI_DEBUG


      //let's now check that we're finished by ensuring we have
      //accounted for all the neighbors
      flag = 0;
      for(j = 0; j < mpi_size_; j++) {
	if(unsorted_counts[j] != 0) {
	  flag = 1;
	  break;
	}
      }
      if(!flag)
	break;
    }
  }

  //now we distribute the results
  MPI_Scatter(sorted_counts, 1, MPI_UNSIGNED, &mpi_neighbor_count_,
	      1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  MPI_Scatter(sorted_neighbors, mpi_size_, MPI_INT, mpi_neighbors_,
	      mpi_size_, MPI_INT, 0, MPI_COMM_WORLD);

  //truncate results
  mpi_neighbors_ = (int*) realloc(mpi_neighbors_, mpi_neighbor_count_ * sizeof(int));

  if(mpi_rank_ == 0) {     

    free(unsorted_neighbors);
    free(sorted_neighbors);
    free(sorted_counts);
    free(unsorted_counts);
    free(b_paired);

  } 
}

void EDM::EDMBias::update_height(edm_data_t bias_added) {
  edm_data_t other_bias = 0;
#ifndef EDM_SERIAL
  MPI_Allreduce(&bias_added, &other_bias, 1, MPI_EDM_DATA_T, MPI_SUM, MPI_COMM_WORLD);
#else
  other_bias = bias_added;
#endif
  cum_bias_ += other_bias;
		
}


void EDM::EDMBias::set_mask(const int* mask) {
  mask_ = mask;
}

int EDM::EDMBias::read_input(const std::string& input_filename){
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
  bias_dx_ = (edm_data_t*) malloc(sizeof(edm_data_t) * dim_);
  bias_sigma_ = (edm_data_t*) malloc(sizeof(edm_data_t) * dim_);
  min_ = (edm_data_t*) malloc(sizeof(edm_data_t) * dim_);
  max_ = (edm_data_t*) malloc(sizeof(edm_data_t) * dim_);
  b_periodic_boundary_ = (int*) malloc(sizeof(int) * dim_);
  
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
  } else {
    b_targeting_ = 1;
    string tfilename = parsed_input.at("target_filename");
    string cleaned_filename = clean_string(tfilename, 0);
    target_ = read_grid(dim_, cleaned_filename, 0); //read grid, do not use interpolation
    expected_target_ = target_->expected_bias();
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


std::string EDM::EDMBias::clean_string(const std::string& input, int append_rank) {
  std::string result(input);
  //remove surrounding whitespace 
  size_t found = result.find_first_not_of(" \t");
  if (found != std::string::npos)
    result = result.substr(found);    
  if(append_rank) {
    std::ostringstream oss;
    oss << result << "_" << mpi_rank_;
    return oss.str();
  }
  return result;

}
