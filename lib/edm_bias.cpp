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

#ifndef GAUSS_SUPPORT
#define GAUSS_SUPPORT 6.25
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
						      buffer_i_(0){
  
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


void EDMBias::subdivide(const double sublo[3], 
			const double subhi[3], 
			const int b_periodic[3],
			const double skin[3]) {

#ifdef EDM_MPI_DEBUG
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(-1 == 0) {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }
  

#endif


  //has subdivide already been called?
  if(bias_ != NULL)
    return;
  
  int grid_period[] = {0, 0, 0};
  double min[3];
  double max[3];
  size_t i;
  int size;
  int bounds_flag = 1;

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
  
  
  bias_ = make_gauss_grid(dim_, min, max, bias_dx_, grid_period, 1, bias_sigma_);
  bias_->set_boundary(min_, max_, b_periodic);

#ifndef SERIAL_TEST
  infer_neighbors(b_periodic, skin);
  //make hill density a per-system measurement not per replica

  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if(hill_density_ > 0) 
    hill_density_ /= size;

  //We have two comm cases, global and neighbors. In global, we build
  //a broadcast tree so that the number of comms is logrithmic in the
  //number of comms. That is with broadcast. If instead it's more
  //efficient to use neighbor communication, then we need to sort our neighbors
  if(mpi_neighbor_count_ < log(size)) {
    //  if(1)
    sort_neighbors();
    std::cout << "Using neighbors" << std::endl;
  } else{
    mpi_neighbor_count_ = size; //just communicate with all
    std::cout << "Using broadcast" << std::endl;
  }


  
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
#ifdef EDM_MPI_DEBUG
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  std::ostringstream oss;
  oss << output << "_" << rank;
  bias_->write(oss.str());
#endif //EDM_DEBUG
#else //SERIAL TEST
  bias_->write(output);
  #endif
}

void EDMBias::setup(double temperature, double boltzmann_constant) {

  temperature_ = temperature;
  boltzmann_factor_ = boltzmann_constant * temperature;

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
	      send_buffer_[buffer_i_ * (dim_+1) + j] = positions[i][j];
	    send_buffer_[buffer_i_ * (dim_+1) + j] = this_h;

	    buffer_i_++;

	    //do we need to flush?
	    if((buffer_i_ + 1) * (dim_+1) >= BIAS_BUFFER_SIZE)
	      bias_added += flush_buffers(0); //flush and we don't know if we're synched

	  }

	  /*	  
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
	  */
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
    
    size_t i,j;
    unsigned int buffer_j;
    int rank, size, result;
    MPI_Request srequest1, srequest2;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(mpi_neighbor_count_ == size) {
      for(i = 0; i < size; i++) {
	if(rank == i) {
	  MPI_Bcast(&buffer_i_, 1, MPI_UNSIGNED, i, MPI_COMM_WORLD);
	  MPI_Bcast(send_buffer_, buffer_i_ * (dim_ + 1), MPI_DOUBLE, i, MPI_COMM_WORLD);
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
      
      for(i = 0; i < mpi_neighbor_count_; i++) {

	if(mpi_neighbors_[i] == NO_COMM_PARTNER)
	  continue;

	//async send, since we don't care about the buffer
	MPI_Isend(&buffer_i_, 1, MPI_UNSIGNED, 
		  mpi_neighbors_[i], i, MPI_COMM_WORLD, 
		  &srequest1);

	MPI_Isend(send_buffer_, buffer_i_ * (dim_ + 1), 
		  MPI_DOUBLE, mpi_neighbors_[i], 
		  i, MPI_COMM_WORLD, &srequest2);

	//do sync receive, because we need it to continue
	MPI_Recv(&buffer_j, 1, MPI_UNSIGNED, 
		 mpi_neighbors_[i], i, 
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Recv(receive_buffer_, buffer_j * (dim_ + 1), 
		 MPI_DOUBLE, mpi_neighbors_[i], 
		 i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

	for(j = 0; j < buffer_j; j++) {
	  bias_added += bias_->add_gaussian(&receive_buffer_[j * (dim_+1)], 
					    receive_buffer_[j * (dim_+1) + dim_]);	
	}

	//Technically, I don't need to do this here but 
	//it's more convienent than having a bunch of outstanding requests
	MPI_Wait(&srequest1, MPI_STATUS_IGNORE);
	MPI_Wait(&srequest2,  MPI_STATUS_IGNORE);

      }
    }

    //reset buffer
    buffer_i_ = 0;
  }


  
  return bias_added;
}

void EDMBias::infer_neighbors(const int* b_periodic, const double* skin) {

   //now the hard part, we need to infer the domain decomposition topology
   size_t i,j;
   int rank, size;
   double* bounds = (double*) malloc(sizeof(double) * dim_ * 2);
   int dim_overlap; //if somethig overlaps in all dimensions
   

   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   MPI_Comm_size(MPI_COMM_WORLD,&size);

   if(mpi_neighbors_ != NULL)
     free(mpi_neighbors_);

   mpi_neighbors_ = (int*) malloc(sizeof(int) * size);
   for(i = 0; i < size; i++) 
     mpi_neighbors_[i] = NO_COMM_PARTNER;

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

   #ifdef EDM_MPI_DEBUG   
   //print out the unsorted neighbors
   for(i = 0; i < size; i++) {
     if(rank == i) {
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
   #endif
 }


/** This method will take the unordered list of neighbors and create sorted versions
 * so that there will be no communication blocks
 */
void EDMBias::sort_neighbors() {

  int* unsorted_neighbors; 
  unsigned int* unsorted_counts;
  int* sorted_neighbors; 
  unsigned int* sorted_counts;//this is include NO_COMM_PARTNER neighbors, so is generally higher
  unsigned int i,j,k,l;
  int* b_paired;
  int rank, size, flag;

   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   MPI_Comm_size(MPI_COMM_WORLD,&size);

   //only execute on head node, because it's too much comm to try to parallelize it
   if(rank == 0) {
     
     sorted_neighbors = (int*) malloc(sizeof(int) * size * size);
     unsorted_neighbors = (int*) malloc(sizeof(int) * size * size);
     sorted_counts = (unsigned int*) malloc(sizeof(unsigned int) * size);
     unsorted_counts = (unsigned int*) malloc(sizeof(unsigned int) * size);
     b_paired = (int*) malloc(sizeof(int) * size);
     
     for(i = 0; i < size; i++)
       sorted_counts[i] = 0;

   }

   MPI_Gather(mpi_neighbors_, size, MPI_INT, unsorted_neighbors, size, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Gather(&mpi_neighbor_count_, 1, MPI_UNSIGNED, unsorted_counts, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

   if(rank == 0) {
     
     for(i = 0; i < size; i++) {//for each round of communication with neighbors, at most size size
       //clear paired flags
       for(j = 0; j < size; j++)
	 b_paired[j] = 0;

       for(j = 0; j < size; j++) {//for each node
	 if(!b_paired[j]) {

	   //assume we won't find a neighbor
	   sorted_neighbors[j * size + sorted_counts[j]] = NO_COMM_PARTNER;
	   
	   //iterate through my neighbor list
	   for(k = 0; k < size && unsorted_neighbors[j * size + k] != NO_COMM_PARTNER; k++) {
	     //NO_COMM_PARTNER happens to mark the end of the array

	     if(!b_paired[unsorted_neighbors[j * size + k]]) { //found an unpaired neighbor!

	       //make sure we haven't already paired with it once before
	       flag = 0;
	       for(l = 0; l < sorted_counts[j]; l++) {
		 if(sorted_neighbors[j * size + l] == unsorted_neighbors[j * size + k]) {
		   flag = 1;
		 break;
		 }
	       }

	       if(!flag) {

		 l = unsorted_neighbors[j * size + k];

		 sorted_neighbors[j * size + sorted_counts[j]] = l;
		 unsorted_counts[j]--;

		 sorted_neighbors[l * size + sorted_counts[l]] = j;
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
       for(j = 0; j < size; j++) {
	 std::cout << j << ": [";
	 for(k = 0; k < sorted_counts[j]; k++)
	   std::cout << sorted_neighbors[j * size + k] << ", ";
	 std::cout << "] <--> [";
	 for(k = 0; k < unsorted_counts[j]; k++)
	   std::cout << unsorted_neighbors[j * size + k] << ", ";
	 std::cout << "]" << std::endl;
	 
     }
#endif// EDM_MPI_DEBUG


       //let's now check that we're finished by ensuring we have
       //accounted for all the neighbors
       flag = 0;
       for(j = 0; j < size; j++) {
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

   MPI_Scatter(sorted_neighbors, size, MPI_INT, mpi_neighbors_,
	       size, MPI_INT, 0, MPI_COMM_WORLD);

   //truncate results
   mpi_neighbors_ = (int*) realloc(mpi_neighbors_, mpi_neighbor_count_ * sizeof(int));

   if(rank == 0) {     

     free(unsorted_neighbors);
     free(sorted_neighbors);
     free(sorted_counts);
     free(unsorted_counts);
     free(b_paired);

   } 
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
