#ifndef EDM_BIAS_H_
#define EDM_BIAS_H_

#include "gaussian_grid.h"
#include "grid.h"
#include "edm.h"
#include <string>
#include <iterator>
#include <sstream>
#include <iostream>
#include <map>


#define BIAS_CLAMP 1000.0
//this one's for mpi:
#define BIAS_BUFFER_SIZE 2048
#define BIAS_BUFFER_DBLS (2048 * 8)
#define NO_COMM_PARTNER -1
#define INTERPOLATE 1

#define NEIGH_HILL 'n'
#define BUFF_HILL 'b'
#define BUFF_UNDO_HILL 'v'
#define ADD_HILL 'h'
#define ADD_UNDO_HILL 'u'
#define BUFF_ZERO_HILL 'z'

namespace EDM{ 

class EDMBias {
  /** The EDM bias class. The main biasing class
   *
   *
   */
 public:

  EDMBias(const std::string& input_filename);
  ~EDMBias();
  /** Create a grid that only occupies enough space for this processes local box.
   * MUSt CALL SETUP FIRST
   *
   **/
  void subdivide(const double sublo[3], const double subhi[3], 
		 const double boxlo[3], const double boxhi[3],
		 const int b_periodic[3], const double skin[3]);


  /** This must be called so that EDM can learn the temperature and kt
   *
   **/
  void setup(double temperature, double boltzmann_constant);
    
  int read_input(const std::string& input_filename);
  /**
   * This version of update_forces is the most general and will update
   * the forces of the arrays. apply_mask will be used to test against
   * the mask given in the set_mask method. Returns energy
   **/
  double update_forces(int nlocal, const double* const* positions, double** forces, int apply_mask) const;
  /**
   * An array-based update_forces without a mask
   **/
  double update_forces(int nlocal, const double* const* positions,  double** forces) const;
  /**
   * Update the force of a single position
   **/
  double update_force(const double* positions,  double* forces) const;
  /**
   * Set a mask that will be used for the add_hills/update_forces methods which can take a mask
   **/
  void set_mask(const int* mask); //set a mask
  /**
   * Add hills using arrays. A precomputed array (different each time) needs to be
   * passed if stochastic sampling is done (otherwise NULL).
   **/
  void add_hills(int nlocal, const double* const* positions, const double* runiform);
  /**
   * Add hills using arrays and taking a mask.
   **/
  void add_hills(int nlocal, const double* const* positions, const double* runiform, int apply_mask);

  /**
   * A way to add hills one at a time. Call pre_add_hill first,
   * add_hill a fixed number of times, and finally post_add_hill. 
   * 
   * It's important to know how many hills will be added and call
   * add_hill that many times.  Times called is the estimated number
   * of times add hill will be called, so it should be the same number
   * each time wihtin a pre/add/post cycle.
   *
   **/
  void pre_add_hill(int est_hill_count);
  void add_hill(const double* position, double runiform);
  void post_add_hill();

  /**
   * Write the bias across all MPI processes. Will also output individual if EDM_MPI_DEBUG is deinfed
   **/
  void write_bias(const std::string& output) const;

  /**
   * Write out histogram of observed points, possibly across multiple processors.
   **/
  void write_histogram() const;

  /**
   * Clear CV histogram
   **/
  void clear_histogram();


  /**
   * Write a lammps table across all MPI processes.  Only valid if
   * we're working with a 1D pairwise-distance potential.
   **/
  void write_lammps_table(const std::string& output) const;


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
  double expected_target_; //the expected value of the target factor

  int b_outofbounds_; //true if this MPI instance will always be out of grid

  double* bias_dx_; //bias spacing
  double* bias_sigma_;
  double* min_; //boundary minimum
  double* max_; //boundary maximum
  int* b_periodic_boundary_;//true if the boundaries are periodic
 

  Grid* target_; //target PMF
  Grid* initial_bias_; //Initial PMF
  GaussGrid* bias_;// bias
  const int* mask_;// a mask to use to exclude atoms

  unsigned int mpi_neighbor_count_;
  int* mpi_neighbors_;//who my neighbors are
  
  //buffers for sending and receiving with neighbors
  double send_buffer_[BIAS_BUFFER_DBLS];
  double receive_buffer_[BIAS_BUFFER_DBLS];
  unsigned int buffer_i_;


  std::ofstream hill_output_;//hill writing

 private:
  //these are used for the pre_add_hill, add_hill, post_add_hill sequence 
  double temp_hill_cum_;
  double temp_hill_prefactor_;
  int est_hill_count_;

  Grid* cv_hist_;//Histogram of observed collective variables
  
  //for printing
  int hills_added_;
  long long int steps_;

  //histogram output
  std::string hist_output_;



  /*
   * This code performs the actual adding of hills in the buffer OR it 
   * calls the GPU host method to invoke a GPU kernel to execute the hill add 
   */
  void do_add_hills(const double* buffer, const size_t hill_number, char hill_type);

  /*
   * This function will put the hills into our buffer of hills to 
   * add. The buffer will be flushed either when it's full or at the end
   * of the overall hill-add step.
   */
  void queue_add_hill(const double* position, double this_h);

    
  EDMBias(const EDMBias& that);//just disable copy constructor
  void output_hill(const double* position, double height, double bias_added, char type);
  /* This will update the height, optionally with tempering. It also
   * will reduce across all processes the average height so that we
   * know if global tempering is necessary.
   *
   */
  void update_height(double bias_added);

  /*
   * Find out which other MPI processes I need to communicate with for add_hills
   */
  void infer_neighbors(const int* b_periodic, const double* skin);
  /*
   * Sort my neighbors into a non-blocking schedule
   */
  void sort_neighbors();

  /*
   * These two methods are used to send hills to my neighobrs
   */
  int check_for_flush();
  double flush_buffers(int snyched);

  double do_add_hill(const double* position, double height, int communicate);

  /*
   * Convienence method to stride whitespace from a string.
   */
  std::string clean_string(const std::string& input, int append_rank);


};

}
#endif // EDM_BIAS_H_
