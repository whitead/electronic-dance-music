#include "edm_bias.h"

#include <cmath>
#include <iterator>
#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include "mpi.h"

#ifdef EDM_MPI_DEBUG
#include "unistd.h"
#endif
 

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


EDMBias::EDMBias(const std::string& input_filename) : b_tempering_(0), 
						      b_targeting_(0), 
						      global_tempering_(0), 
						      hill_density_(-1),
						      cum_bias_(0), 
						      b_outofbounds_(0), 
						      target_(NULL), 
						      bias_(NULL), 
						      mask_(NULL),
						      mpi_neighbor_count_(0),
						      mpi_neighbors_(NULL),
						      buffer_i(0){
  
  //read input file
  read_input(input_filename);  
  
}

EDMBias::~EDMBias() {
  if(target_ != NULL)
    delete target_;
  if(bias_ != NULL)
    delete bias_;

  if(mpi_neighbors_ != NULL)
    free(mpi_neighbors_);
}


void EDMBias::subdivide(const double sublo[3], const double subhi[3], const int b_periodic[3]) {

  //has subdivide already been called?
  if(bias_ != NULL)
    return;
  
  int grid_period[] = {0, 0, 0};
  double min[3];
  double max[3];
  size_t i;


  int bounds_flag = 1;
  for(i = 0; i < dim_; i++) {
    //check if we encapsulate the entire bounds in any dimension
    if(fabs(sublo[i] - min_[i]) < 0.000001 && fabs(subhi[i] - max_[i]) < 0.000001) {
      grid_period[i] = b_periodic[i];
      bounds_flag = 0;      
    }
      
    min[i] = sublo[i];      
    max[i] = subhi[i];      

    //check if we'll always be out of bounds
    bounds_flag &= (min[i] >= max_[i] || max[i] <= min_[i]);    
    
  }
  
  
  bias_ = make_gauss_grid(dim_, min, max, bias_dx_, grid_period, 1, bias_sigma_);
  bias_->set_boundary(min_, max_, b_periodic);

#ifndef SERIAL_TEST
  infer_neighbors(b_periodic);
#endif

  if(bounds_flag) {
    //we do this after so that we have a grid to at least write out
    std::cout << "I am out of bounds!" << std::endl;
    b_outofbounds_ = 1;
    return;
  }

  //get volume
  double other_vol = 0;
  double vol = bias_->get_volume();
  total_volume_ = 0;
  #ifndef SERIAL_TEST
  MPI_Allreduce(&vol, &other_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #else
  other_vol = vol;
  #endif
  total_volume_ += other_vol;

}

void EDMBias::write_bias(const std::string& output) const {
  #ifndef SERIAL_TEST
  bias_->multi_write(output);
  #else
  bias_->write(output);
  #endif
}

void EDMBias::setup(double temperature, double boltzmann_constant) {

  temperature_ = temperature;
  boltzmann_factor_ = boltzmann_constant * temperature;

#ifdef EDM_MPI_DEBUG
int rank;
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
if(rank == 0) {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
 }

#endif

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
    if(apply_mask < 0 || mask_[i] & apply_mask) {
      bias_->get_value_deriv(&positions[i][0], der);
      for(j = 0; j < dim_; j++)
	forces[i][j] -= der[j];
    }
  }

}

void EDMBias::add_hills(int nlocal, const double* const* positions, const double* runiform) {
  add_hills(nlocal, positions, runiform, -1);
}

void EDMBias::add_hills(int nlocal, const double* const* positions, const double* runiform, int apply_mask) {

  int i, j;
  double bias_added = 0;
  double h = hill_prefactor_;  
  double this_h;
  int natoms = 0;

  //are we active?
  if(!b_outofbounds_) {



    //get current hill height
    if(global_tempering_ > 0)
      if(cum_bias_ / total_volume_ >= global_tempering_)
	h *= exp(-(cum_bias_ / total_volume_ - 
		   global_tempering_) / 
		 ((bias_factor_ - 1) * boltzmann_factor_));                   

    
    //count how many atoms we have in bounds
    for(i = 0; i < nlocal; i++)
      if(apply_mask > 0 && mask_[i] & apply_mask)
	if(bias_->in_bounds(&positions[i][0]))
	  natoms++;
    
    
    for(i = 0; i < nlocal; i++)  {   
      
      if(apply_mask < 0 || mask_[i] & apply_mask) {
	//actually add hills -> stochastic
	if(hill_density_ < 0 || runiform[i] < hill_density_ / natoms) {    
	  this_h = h; 
	  if(b_targeting_)
	    this_h *= exp(-target_->get_value(&positions[i][0])); // add target
	  if(b_tempering_ && global_tempering_ < 0) //do tempering if local tempering (well) is being used
	    this_h *= exp(-bias_->get_value(&positions[i][0]) / 
			  ((bias_factor_ - 1) * boltzmann_factor_));
	  //finally clamp bias
	  this_h = fmin(this_h, BIAS_CLAMP * boltzmann_factor_);
	  bias_added += bias_->add_gaussian(&positions[i][0], this_h);
	  
	  //pack result into buffer if necessary
	  if(mpi_neighbor_count_ > 0) {
	    for(j = 0; j < dim_; j++)	    
	      send_buffer_[buffer_i * (dim_+1) + j] = positions[i][j];
	    send_buffer_[buffer_i * (dim_+1) + j] = this_h;

	    buffer_i++;

	    //do we need to flush?
	    if(buffer_i * (dim_+1) == BIAS_BUFFER_SIZE)
	      bias_added += flush_buffers(0); //flush and we don't know if we're synched

	  }
	  
	  //output info/*
	  std::cout << "|- " << bias_added / sqrt(2 * M_PI) / bias_sigma_[0] 
		    << " (" << h << "*";
	  if(b_targeting_)
	    std::cout << "exp(" 
		      << target_->get_value(&positions[i][0]) 
		      << ") ->" 
		      << exp(-target_->get_value(&positions[i][0]));
	  else
	    std::cout << 1;
	  std::cout << ") "
		    << positions[i][0] 
		    << std::endl;
	}
      }
    }
  }

  bias_added += flush_buffers(0); //flush, but we don't know if we're synched

  //some processors may have had to flush more than once if there are
  //lots of hills, continue waiting for them until we all agree no more flush
  while(check_for_flush())
    bias_added += flush_buffers(1); //flush and we are synched

  update_height(bias_added);
}

int EDMBias::check_for_flush() {

  if(mpi_neighbor_count_ > 0) {

    int my_flush = 0;
    int do_flush;
    MPI_Allreduce(&my_flush, &do_flush, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "Flush? -> " << do_flush << std::endl;
    if(do_flush)
      return 1;

  }

  return 0;

}

double EDMBias::flush_buffers(int synched) {

  double bias_added = 0;

  if(mpi_neighbor_count_ > 0) {
    
    //notify all that we're going to flush
    if(!synched) {
      int my_flush = 1;
      int do_flush;
      MPI_Allreduce(&my_flush, &do_flush, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    
    size_t i,j,buffer_j;
    int rank, size, result;
    MPI_Request srequest, rrequest;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    std::cout << "I am " << rank << " and I'm about to begin flush" << std::endl;
    
    
    if(mpi_neighbor_count_ == size) {
      for(i = 0; i < size; i++) {
	if(rank == i) {
	  MPI_Bcast(&buffer_i, 1, MPI_UNSIGNED, i, MPI_COMM_WORLD);
	  MPI_Bcast(send_buffer_, buffer_i * (dim_ + 1), MPI_DOUBLE, i, MPI_COMM_WORLD);
	} else {
	  MPI_Bcast(&buffer_j, 1, MPI_UNSIGNED, i, MPI_COMM_WORLD);
	  MPI_Bcast(receive_buffer_, buffer_j * (dim_ + 1), MPI_DOUBLE, i, MPI_COMM_WORLD);
	  for(j = 0; j < buffer_j; j++) {
	    bias_added += bias_->add_gaussian(&receive_buffer_[j * (dim_+1)], 
					      receive_buffer_[j * (dim_+1) + dim_]);
	  }
	}
      }
    } else {
      
      for(i = 0;i < mpi_neighbor_count_; i++) {

	//start with receive
	MPI_Irecv(&buffer_j, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &rrequest);

	if(mpi_neighbors_[i] >= 0 && buffer_i > 0) {
	  //want to make sure the send is complete before we get to barrier
	  MPI_Send(&buffer_i, 1, MPI_UNSIGNED, mpi_neighbors_[i], i, MPI_COMM_WORLD);
	  std::cout << "I am " << rank 
		    << " and I'm sending " 
		    << buffer_i 
		    << " items to " 
		    << mpi_neighbors_[i] 
		    << std::endl;
	} else {
	  std::cout << "I am " << rank << " and won't send " << std::endl;
	}
	
	//now make sure everyone is done before we send/receive buffers so we know for sure who is getting one
	MPI_Barrier(MPI_COMM_WORLD);
	
	//now send buffer if needed
	if(mpi_neighbors_[i] >= 0 && buffer_i > 0) {
	  MPI_Isend(send_buffer_, buffer_i * (dim_ + 1), MPI_DOUBLE, mpi_neighbors_[i], i, MPI_COMM_WORLD, &srequest); //no blocking
	}
	
	//if we did get a receive, we know we have an incoming buffer and we need to finish the process
	MPI_Test(&rrequest, &result, MPI_STATUS_IGNORE);
	if(result) {
	  std::cout << "I am " << rank << " and I'm about to get " << buffer_j << std::endl;
	  MPI_Recv(receive_buffer_, buffer_j * (dim_ + 1), MPI_DOUBLE, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //note we block here
	  for(j = 0; j < buffer_j; j++) {
	    bias_added += bias_->add_gaussian(&receive_buffer_[j * (dim_+1)], receive_buffer_[j * (dim_+1) + dim_]);	
	  }
	} else {
	  //clean up operation
	  MPI_Cancel(&rrequest);	  
	  MPI_Request_free(&rrequest);    
	  std::cout << "I am " << rank << " and no one wants to talk to me " << std::endl;
	}

	//finally wait for send to finish, if we did send
	if(mpi_neighbors_[i] >= 0 && buffer_i > 0)
	  MPI_Wait(&srequest, MPI_STATUS_IGNORE);

      }
    }

    //reset buffer
    buffer_i = 0;
    std::cout << "I am " << rank << " and I ended up with  " << bias_added << "new bias" << std::endl;
  }    


  
  return bias_added;
}

 void EDMBias::infer_neighbors(const int* b_periodic) {

   //now the hard part, we need to infer the domain decomposition topology
   size_t i,j;
   int rank, size;
   double* bounds = (double*) malloc(sizeof(double) * dim_ * 2);
   

   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   MPI_Comm_size(MPI_COMM_WORLD,&size);

   if(mpi_neighbors_ != NULL)
     free(mpi_neighbors_);

   mpi_neighbors_ = (int*) malloc(sizeof(int) * size);
   for(i = 0; i < size; i++) 
     mpi_neighbors_[i] = -1;

   for(i = 0; i < size; i++) {   //for each rank 

     if(rank == i) {//it's my turn to broadcast my boundaries

       for(j = 0; j < dim_; j++) {//pack up my bounds
	 bounds[j*2] = bias_->get_min()[j];
	 bounds[j*2 + 1] = bias_->get_max()[j];
       }
       MPI_Bcast(bounds, dim_ * 2, MPI_DOUBLE, i, MPI_COMM_WORLD);

     }  else {
       MPI_Bcast(bounds, dim_ * 2, MPI_DOUBLE, i, MPI_COMM_WORLD);
       //check if this could be a neighbor
       for(j = 0; j < dim_; j++) {
	 if(bias_->get_max()[j] + 6 * bias_sigma_[j] > bounds[j*2] &&
	    bias_->get_max()[j] + 6 * bias_sigma_[j] < bounds[j*2 + 1]) {
	   mpi_neighbors_[mpi_neighbor_count_] = i;
	   mpi_neighbor_count_++;
	   break;
	 } else if(bias_->get_min()[j] - 6 * bias_sigma_[j] < bounds[j * 2 + 1] &&
		   bias_->get_min()[j] - 6 * bias_sigma_[j] > bounds[j * 2]) {
	   mpi_neighbors_[mpi_neighbor_count_] = i;
	   mpi_neighbor_count_++;
	   break;
	 }
	 //those were the easy cases, now wrapping
	 if(b_periodic[j]) {
	   
	   if((fabs(bias_->get_min()[j] - min_[j]) < 6 * bias_sigma_[j] && // I am at left
	       fabs(bounds[j * 2 + 1] - max_[j] - bias_dx_[j]) < 6 * bias_sigma_[j]) || //other is at right
	      (fabs(bias_->get_max()[j] - max_[j] - bias_dx_[j]) < 6 * bias_sigma_[j] && //or I am at right
	       fabs(bounds[j * 2] - min_[j]) < 6 * bias_sigma_[j])) {//other is at left
	     mpi_neighbors_[mpi_neighbor_count_] = i;
	     mpi_neighbor_count_++;
	     break;	     
	   }	   	   
	 }	 
       }
     }
   }

   //now, we need to find the maximum neighbor number so we can synchronize our hill passing
   unsigned int temp = mpi_neighbor_count_;
   MPI_Allreduce(&temp, &mpi_neighbor_count_, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);

   mpi_neighbors_ = (int*) realloc(mpi_neighbors_, mpi_neighbor_count_ * sizeof(int));

   for(i = 0; i < size; i++) {
     if(rank == i) {
       std::cout << "Neighobrs of " << i << "== ";
       for(j = 0; j < mpi_neighbor_count_; j++){
	 std::cout << mpi_neighbors_[j] << " ";
       }
       std::cout << std::endl;
     }
   }
   std::cout << std::endl;
 }

void EDMBias::update_height(double bias_added) {
  double other_bias = 0;
  #ifndef SERIAL_TEST
  MPI_Allreduce(&bias_added, &other_bias, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #else
  other_bias = bias_added;
  #endif
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
    b_targeting_ = 0;
  } else {
    b_targeting_ = 1;
    string& tfilename = parsed_input.at("target_filename");
    //remove surrounding whitespace 
    size_t found = tfilename.find_first_not_of(" \t");
    if (found != string::npos)
      tfilename = tfilename.substr(found);
    target_ = read_grid(dim_, tfilename, 0); //read grid, do not use interpolation
  }
 
  return 1;
}
