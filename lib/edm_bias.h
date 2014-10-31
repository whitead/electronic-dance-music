#ifndef EDM_BIAS_H_
#define EDM_BIAS_H_

#include "gaussian_grid.h"
#include "grid.h"
#include <string>
#include <iterator>
#include <sstream>
#include <iostream>
#include <map>

#define BIAS_CLAMP 10
//number of doubles
#define BIAS_BUFFER_SIZE 64
#define NO_COMM_PARTNER -1


class EDMBias {
  /** The EDM bias class.
   *
   *
   */
 public:
  
  EDMBias(const std::string& input_filename);
  ~EDMBias();
  /** Create a grid that only occupies enough space for this processes local box
   *
   */
  void subdivide(const double sublo[3], const double subhi[3], 
		 const double boxlo[3], const double boxhi[3],
		 const int b_periodic[3], const double skin[3]);


  void setup(double temperature, double boltzmann_constant);
    
  int read_input(const std::string& input_filename);
  void update_forces(int nlocal, const double* const* positions, double** forces, int apply_mask) const;
  void update_forces(int nlocal, const double* const* positions,  double** forces) const;
  void update_force(const double* positions,  double* forces) const;
  void set_mask(const int* mask); //set a mask
  /**
   * Add hills. A precomputed array (different each time) needs to be
   * passed if stochastic sampling is done (otherwise NULL).
   **/
  void add_hills(int nlocal, const double* const* positions, const double* runiform);
  void add_hills(int nlocal, const double* const* positions, const double* runiform, int apply_mask);

  /**
   * A way to add hills one at a time. Call pre_add_hill first,
   * add_hill a fixed number of times, and finally post_add_hill. 
   * 
   * It's important to know how many hills will be added and call add_hill that many times.
   *
   **/
  void pre_add_hill();
  void add_hill(int times_called, const double* position, double runiform);
  void post_add_hill();

  void write_bias(const std::string& output) const;


  int b_tempering_;// boolean, do tempering
  int b_targeting_;// boolean, do targeting  
  int mpi_rank_; //my MPI rank
  int mpi_size_; //my MPI size
  unsigned int dim_;//the dimension
  double global_tempering_;// global tempering threshold
  double bias_factor_;
  double boltzmann_factor_;
  double temperature_;
  double hill_prefactor_; //hill height prefactor
  double hill_density_;// hills sampling density
  double cum_bias_;//the current average bias  
  double total_volume_;//total volume of grid 

  int b_outofbounds_; //true if this MPI instance will always be out of grid

  double* bias_dx_; //bias spacing
  double* bias_sigma_;
  double* min_; //boundary minimum
  double* max_; //boundary maximum
 

  Grid* target_; //target PMF
  GaussGrid* bias_;// bias
  const int* mask_;// a mask to use to exclude atoms

  unsigned int mpi_neighbor_count_;
  int* mpi_neighbors_;//who my neighbors are
  
  //buffers for sending and receiving with neighbors
  double send_buffer_[BIAS_BUFFER_SIZE];
  double receive_buffer_[BIAS_BUFFER_SIZE];
  unsigned int buffer_i_;

  std::ofstream hill_output_;

 private:
  //these are used for the pre_add_hill, add_hill, post_add_hill sequence 
  double temp_hill_cum_;
  double temp_hill_prefactor_;
  
  EDMBias(const EDMBias& that);
  void output_hill(const double* position, double height, double bias_added);
  /** This will update the height, optionally with tempering. It also
   * will reduce across all processes the average height so that we
   * know if global tempering is necessary.
   *
   */
  void update_height(double bias_added);

  void infer_neighbors(const int* b_periodic, const double* skin);
  void sort_neighbors();

  int check_for_flush();
  double flush_buffers(int snyched);

  std::string clean_string(const std::string& input, int append_rank);


};

#endif // EDM_BIAS_H_
