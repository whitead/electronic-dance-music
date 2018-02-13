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

namespace std {
  extern istream& operator >> (istream& is, pair<string, string>& ps);
  
  extern ostream& operator << (ostream& os, const pair<const string, string>& ps);
  
}


namespace EDM{ 

  extern int extract_edm_data_t(const std::string& key, std::map<std::string, std::string> map, int required, edm_data_t* result);

  extern int extract_edm_data_t_array(const std::string& key, std::map<std::string, std::string> map, int required, edm_data_t* result, int length);

  extern int extract_int(const std::string& key, std::map<std::string, std::string> map, int required, int* result);
  
  
  class EDMBias {
    /** The EDM bias class. The main biasing class
     *
     *
     */
  public:

    EDMBias(const std::string& input_filename);
    ~EDMBias();
    /** Create a grid that only occupies enough space for this process' local box.
     * MUSt CALL SETUP FIRST
     *
     **/
    void subdivide(const edm_data_t sublo[3], const edm_data_t subhi[3], 
		   const edm_data_t boxlo[3], const edm_data_t boxhi[3],
		   const int b_periodic[3], const edm_data_t skin[3]);


    /** This must be called so that EDM can learn the temperature and kt
     *
     **/
    void setup(edm_data_t temperature, edm_data_t boltzmann_constant);
    
    virtual int read_input(const std::string& input_filename);
    /**
     * This version of update_forces is the most general and will update
     * the forces of the arrays. apply_mask will be used to test against
     * the mask given in the set_mask method. Returns energy
     **/
    edm_data_t update_forces(int nlocal, const edm_data_t* const* positions, edm_data_t** forces, int apply_mask) const;
    /**
     * An array-based update_forces without a mask
     **/
    edm_data_t update_forces(int nlocal, const edm_data_t* const* positions,  edm_data_t** forces) const;
    /**
     * Update the force of a single position
     **/
    edm_data_t update_force(const edm_data_t* positions,  edm_data_t* forces) const;
    /**
     * Set a mask that will be used for the add_hills/update_forces methods which can take a mask
     **/
    void set_mask(const int* mask); //set a mask
    /**
     * Add hills using arrays. A precomputed array (different each time) needs to be
     * passed if stochastic sampling is done (otherwise NULL).
     **/
    virtual void add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform);
    /**
     * Add hills using arrays and taking a mask.
     **/
    virtual void add_hills(int nlocal, const edm_data_t* const* positions, const edm_data_t* runiform, int apply_mask);

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
    void add_hill(const edm_data_t* position, edm_data_t runiform);
    virtual void post_add_hill();

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
    int dim_;//the dimension
    edm_data_t global_tempering_;// global tempering threshold
    edm_data_t bias_factor_;
    edm_data_t boltzmann_factor_;
    edm_data_t temperature_;
    edm_data_t hill_prefactor_; //hill height prefactor
    edm_data_t hill_density_;// hills sampling density
    edm_data_t cum_bias_;//the current average bias  
    edm_data_t total_volume_;//total volume of grid
    edm_data_t expected_target_; //the expected value of the target factor

    int b_outofbounds_; //true if this MPI instance will always be out of grid

    edm_data_t* bias_dx_; //bias spacing
    edm_data_t* bias_sigma_;
    edm_data_t* min_; //boundary minimum
    edm_data_t* max_; //boundary maximum
    int* b_periodic_boundary_;//true if the boundaries are periodic
 

    Grid* target_; //target PMF
    Grid* initial_bias_; //Initial PMF
    GaussGrid* bias_;// bias
    const int* mask_;// a mask to use to exclude atoms

    unsigned int mpi_neighbor_count_;
    int* mpi_neighbors_;//who my neighbors are
  
    //buffers for sending and receiving with neighbors
    edm_data_t send_buffer_[BIAS_BUFFER_DBLS];
    edm_data_t receive_buffer_[BIAS_BUFFER_DBLS];
    unsigned int buffer_i_;


    std::ofstream hill_output_;//hill writing

    /*
     * Convienence method to stride whitespace from a string.
     */
    std::string clean_string(const std::string& input, int append_rank);

  protected:
    //these are used for the pre_add_hill, add_hill, post_add_hill sequence 
    edm_data_t temp_hill_cum_;
    edm_data_t temp_hill_prefactor_;
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
    virtual edm_data_t do_add_hills(const edm_data_t* buffer, const size_t hill_number, char hill_type);

    /*
     * This function will put the hills into our buffer of hills to 
     * add. The buffer will be flushed either when it's full or at the end
     * of the overall hill-add step.
     */
    virtual void queue_add_hill(const edm_data_t* position, edm_data_t this_h);

    
    EDMBias(const EDMBias& that);//just disable copy constructor
    void output_hill(const edm_data_t* position, edm_data_t height, edm_data_t bias_added, char type);
    /* This will update the height, optionally with tempering. It also
     * will reduce across all processes the average height so that we
     * know if global tempering is necessary.
     *
     */
    void update_height(edm_data_t bias_added);

    /*
     * Find out which other MPI processes I need to communicate with for add_hills
     */
    void infer_neighbors(const int* b_periodic, const edm_data_t* skin);
    /*
     * Sort my neighbors into a non-blocking schedule
     */
    void sort_neighbors();

    /*
     * These two methods are used to send hills to my neighobrs
     */
    int check_for_flush();
    virtual edm_data_t flush_buffers(int snyched);



  };

}
#endif // EDM_BIAS_H_
